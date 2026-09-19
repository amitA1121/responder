# satrecon — System Overview & Handoff

A complete, self-contained description of the `satrecon` pipeline: what it is,
how it is built, how to run it, the data it produces, and the concrete work
still open. Written so that a developer or an AI agent can pick the project up
cold, without reading every source file first.

---

## 1. What the system is

`satrecon` turns a **single satellite or aerial image** into an **approximate,
editable 3D model** of the buildings visible in it. It runs fully locally, needs
no network, no GPU, and no cloud service. Its only runtime dependencies are
NumPy, OpenCV, Pillow and PyYAML.

The pipeline has two halves:

1. **Observation** — from the image it detects buildings, extracts clean
   footprint polygons, estimates the illumination (shadow) direction, resolves
   scale, resolves the sun position, and estimates each building's height and
   floor count from the length of its cast shadow.
2. **Generation** — it extrudes the reviewed footprints to their estimated
   heights into a watertight 3D mesh and exports it as GLB, glTF or OBJ.

Everything the system produces is an **estimate derived from pixels**, carrying a
`source` and a `confidence`. It is not survey data. The design goes to some
length to make that honest: it never invents a scale or a sun elevation, and any
quantity it cannot support from the user's inputs stays `null` rather than being
guessed.

### Intended use

Reconstructing rough 3D city/site models from overhead imagery: urban-planning
massing studies, real-estate and GIS context models, game/VFX greyboxing,
disaster-response damage context, and as a teaching example of classical CV +
photogrammetry. Accuracy is meant to be validated against open ground-truth
data (e.g. OpenStreetMap building footprints and `building:levels`).

---

## 2. High-level flow

```
 image
   │
   ▼
 preprocess ─▶ surface cues ─▶ shadow direction ─▶ segmentation
                                                       │
                                                       ▼
              footprints ◀── detection (pluggable) ◀───┘
                   │
                   ▼
           scale ─▶ sun ─▶ height (shadow rays)
                   │
                   ▼
             scene JSON  ──▶  review.html (2D check)
                   │
                   ▼
          3D extrude ─▶ GLB / glTF / OBJ
```

Each arrow is a module boundary. Stages do not reach across one another; the
pipeline file only wires them together and records provenance. The 3D
generation runs **separately** from analysis, driven from a saved scene, so a
human (or another tool) can correct footprints/heights before geometry is built.

---

## 3. Repository layout

```
src/satrecon/
  __init__.py            # version, MODEL_VERSION, PHASE constants
  __main__.py            # `python -m satrecon`
  cli.py                 # argparse CLI: analyze / review / generate / verify / detectors
  config.py              # DEFAULTS + YAML loading + dotted lookup + fingerprint
  imaging.py             # load, hash, downscale-to-working, coord mapping
  logging_setup.py       # logger configuration
  model.py               # data model + scene JSON (de)serialization + result_hash
  pipeline.py            # orchestration: analyze() runs the 8 stages
  detect/
    base.py              # BuildingDetector ABC, DetectionContext, DetectorResult
    registry.py          # register_detector / get_detector / available_detectors
    classical.py         # the built-in classical CV detector ("classical")
  stages/
    preprocess.py        # (1) mean-shift smoothing
    cues.py              # (2) vegetation/shadow/soil/texture/lightness masks
    shadow.py            # (3) dominant shadow direction + adjacency map
    segmentation.py      # (4) gradient watershed into regions
    regionfeatures.py    # per-region shape & radiometry (used by classical)
    footprint.py         # (5) simplify/regularize contours -> Building polygons
    scale.py             # (6) resolve metres-per-pixel from user input only
    sun.py               # (7) solar elevation/azimuth (NOAA) or stated
    height.py            # (8) shadow-ray height + floor estimation
  geometry/
    mesh.py              # extrude footprints -> watertight prisms (ear clipping)
    export.py            # GLB / glTF / OBJ writers (dependency-free)
  debug/
    render.py            # overlay + stage rasters
    review.py            # interactive review.html generator
tests/                   # pytest suite (see §11)
config/site.example.yaml # annotated config template
docs/architecture.md     # design rationale (companion to this file)
docs/SYSTEM_OVERVIEW.md  # this file
input/  data/  output/   # image in / scene JSON / rendered artifacts (gitignored)
```

---

## 4. The eight analysis stages

| # | Stage | Module | Job | Key output |
|---|-------|--------|-----|------------|
| 1 | Preprocess | `stages/preprocess.py` | Edge-preserving mean-shift smoothing; flattens roof texture, keeps boundaries | smoothed BGR + source LAB |
| 2 | Cues | `stages/cues.py` | Per-pixel masks: vegetation (excess-green), shadow (low L), soil (chroma), texture, lightness | `CueMaps` |
| 3 | Shadow direction | `stages/shadow.py` | Angle from a structure to its own shadow, by aligning the shadow mask with bright surfaces over all angles/distances; confidence = peakedness | `ShadowDirection(angle_deg, confidence, …)` |
| 4 | Segmentation | `stages/segmentation.py` | Gradient watershed seeded on low-gradient plateaus so region borders follow real roof edges | labelled regions |
| 5 | Detection | `detect/classical.py` (pluggable) | Score each region on brightness/rectangularity/solidity/shadow-adjacency/texture/thickness minus vegetation/shadow/soil/elongation penalties; merge adjacent roof facets | `BuildingDetection[]` |
| 5b | Footprints | `stages/footprint.py` | Douglas–Peucker simplify, optional axis regularization, map working→source pixels, confidence from detector score + polygon fit | `Building[]` |
| 6 | Scale | `stages/scale.py` | Resolve metres-per-pixel from user input **only** | `ScaleInfo` |
| 7 | Sun | `stages/sun.py` | Solar elevation (+azimuth) from `sun.elevation_deg`, or NOAA algorithm on geo+timestamp, else unknown | `SunPosition` |
| 8 | Height | `stages/height.py` | Cast rays from shadow-facing footprint edges, measure shadow run, `height = shadow_len × tan(elevation)`, derive floors | heights written into `Building[]` |

### Physics that matters

- **Scale** (`stages/scale.py`): resolution order — (1) explicit
  `meters_per_pixel`, (2) two reference points + a known distance, (3) product
  ground-sample-distance, (4) unknown → dimensions stay in pixels. There is **no
  image-only fallback**; a wrong scale silently corrupts every dimension.
- **Sun** (`stages/sun.py`): resolution order — (1) explicit `elevation_deg`,
  (2) NOAA solar-position from `geo.center_lat/lon` + a timezone-aware
  `sun.timestamp`, (3) unknown → heights not estimated. A below-horizon result
  is reported *unknown*, not negative. The computed case also yields the sun
  azimuth, which gives the shadow direction the sun *implies* — surfaced as a
  cross-check against the image-derived direction, never used to override it.
- **Height** (`stages/height.py`): the shadow length is measured, not assumed.
  Rays start on the footprint edges whose outward normal faces the shadow
  direction, march along that direction, and count the shadow run (tolerating
  small gaps). The robust median of the ray lengths is the shadow length; the
  ray spread is an agreement score. Height confidence is **multiplicative**:
  `ray_agreement × shadow_direction_confidence × sun_confidence`, capped further
  when few rays contributed — a precise measurement in a wrong direction is
  still wrong. Floors = `round(height / floor_height_m)`, floored at 1.

---

## 5. The detector plugin interface (the main extension point)

Detection is deliberately swappable. Everything downstream consumes
`BuildingDetection` objects and does not know which detector produced them.

Contract (`detect/base.py`):

```python
class BuildingDetector(ABC):
    name: str = "abstract"
    def __init__(self, params: dict | None = None): ...
    @abstractmethod
    def detect(self, context: DetectionContext) -> DetectorResult: ...
```

- `DetectionContext` offers already-computed inputs (`bgr`, `preprocessed`,
  `cues`, `shadow_direction`, `shadow_adjacency`, `segmentation`). A detector may
  use any of them or work from `bgr` alone.
- `DetectorResult` = `detections: list[BuildingDetection]` + optional
  `debug_layers` + `notes`.
- Coordinates returned are **working-image pixels** (the pipeline maps them back
  to source pixels in the footprint stage).

Register and select by name:

```python
from satrecon.detect import register_detector
register_detector("my_detector", MyDetectorFactory)
# then in config:  detector: { name: my_detector, params: {...} }
```

`BuildingDetection` fields: `id`, `contour` (dense points), `bounding_box`,
`confidence`, `orientation_deg`, `area_px`, `detector`, `evidence` (free-form
dict shown in the review page).

---

## 6. Data model & scene JSON

Defined in `model.py`. `Scene.save()` writes JSON; `Scene.load()` reads it back.

Top-level structure:

```jsonc
{
  "schemaVersion": 1,
  "scene": {
    "name": "...",
    "image": { "path", "sha256", "width", "height" },
    "scale": { "meters_per_pixel", "source", "confidence", "note" },
    "shadowDirectionDeg": 0.0,
    "northOffsetDeg": null,
    "sun": {
      "elevation_deg", "azimuth_deg", "source", "confidence",
      "expected_shadow_direction_deg"
    },
    "notes": [ "...human-readable provenance and caveats..." ]
  },
  "provenance": {
    "toolVersion", "modelVersion", "detector",
    "configFingerprint", "generatedAt", "resultHash"
  },
  "config": { ...effective config used... },
  "buildings": [
    {
      "id": "B001",
      "footprint": [[x,y], ...],     // simplified polygon, SOURCE pixels
      "contour":   [[x,y], ...],     // original dense contour
      "bounding_box": [x,y,w,h],
      "orientation_deg", "area_px", "length_px", "width_px",
      "length_m", "width_m", "area_m2",         // null if scale unknown
      "estimated_height_m", "height_source",    // "shadow" | "not_estimated" | "user_edited"
      "estimated_floors", "floors_source",
      "roof_type": "unknown", "roof_source": "not_inferred",  // Phase 3
      "confidence": {
        "footprint", "scale", "height", "floors",  // null = not estimated this run
        "overall", "band"                          // band: HIGH/MEDIUM/LOW
      },
      "evidence": { ...detector cues + shadow_length_px, shadow_rays, ray_agreement... },
      "user_edited": false
    }
  ]
}
```

### Model invariants (do not break these)

- **`null` ≠ `0.0` in confidence.** `null` = *not estimated this run* and is
  excluded from the average; `0.0` = *estimated and unreliable* and is included.
  Conflating them makes an unknown scale look like a bad footprint.
- **Source data stays separate from generated geometry.** The scene JSON records
  observations; the 3D mesh is a derived artifact, always regenerated from the
  scene, never edited in place.
- **`user_edited: true` is authoritative.** The height stage already refuses to
  recompute a user-edited height. Any future re-run logic must treat such
  records as fixed and rebuild everything else around them.
- **Determinism.** Same image + same config ⇒ same `resultHash` (computed over
  geometry and heights, excluding the timestamp). The one RNG in the codebase
  (debug label colours) is explicitly seeded. `satrecon verify` enforces this.

---

## 7. 3D geometry & export

`geometry/mesh.py`:

- `building_to_mesh(building, scale, default_height_units)` lifts one footprint
  into a **watertight prism**: floor cap, roof cap, wall band. Footprints are
  often concave (L/U/cross), so caps are triangulated by **ear clipping**, not
  by assuming convexity. Coordinates: X east, Z south (image +y), **Y up**.
  Units are metres when scale is known, pixels otherwise. Buildings with no
  measured height get `default_height_units` as a visible placeholder block.
- `scene_to_mesh(buildings, scale, default_height_units=3.0)` merges all
  buildings into one `Mesh(vertices, faces, face_building)`.

`geometry/export.py` (no third-party dependency):

- `write_glb` — binary glTF (12-byte header + JSON chunk + BIN chunk). Units are
  recorded in `asset.extras.units`. Opens in Blender, three.js, and standard
  glTF viewers.
- `write_gltf` — text `.gltf` with a base64 data-URI buffer.
- `write_obj` — Wavefront OBJ, one `g <building_id>` group per building, units in
  a header comment.

---

## 8. CLI reference

Installed entry point `satrecon` (or `python -m satrecon`). Common flags:
`--config/-c <yaml>`, `--log-level`.

| Command | Purpose | Key options |
|---|---|---|
| `analyze <image>` | Run the 8 stages, write scene JSON + overlay + review page | `--out/-o`, `--debug-dir`, `--name`, `--no-debug`, `--no-review`, `--embed-image` |
| `review <scene.json>` | (Re)build the 2D review HTML from a scene | `--out/-o`, `--image`, `--embed-image` |
| `generate <scene.json>` | Build 3D geometry from a reviewed scene and export | `--out-dir/-o`, `--formats glb,obj,gltf`, `--default-height` |
| `verify <scene.json>` | Re-run analysis and compare the result hash (reproducibility) | — |
| `detectors` | List registered detectors | — |

Typical output paths: `data/<name>.json`, `output/<name>/overlay.png`,
`output/<name>/review.html`, `output/<name>/model/<name>.glb`.

---

## 9. Configuration reference

Defaults live in `config.py::DEFAULTS`; a YAML file supplies overrides and,
crucially, the **facts the user knows** (scale, sun, coordinates, north offset).
User-supplied values are authoritative and are never overwritten by a heuristic.
Sections:

- `image.north_offset_deg` — anticlockwise degrees from image-up to true north
  (null ⇒ orientations reported image-relative).
- `scale.{meters_per_pixel | reference{point_a,point_b,distance_meters} | ground_sample_distance}`.
- `geo.{center_lat, center_lon, crs}`.
- `sun.{elevation_deg | azimuth_deg | timestamp}` — timestamp must be ISO 8601
  **with a UTC offset**.
- `height.floor_height_m` and ray-marching params
  (`ray_spacing_px`, `max_shadow_px`, `gap_tolerance_px`, `start_offset_px`, `min_rays`).
- `preprocess`, `cues`, `shadow_direction`, `segmentation` — stage tuning.
- `detector.{name, params}` — selects and configures the detector. For
  `classical`, the most-tuned knob is `params.accept_score` (lower = more
  detections + more false positives; higher = fewer, higher precision).
- `footprint.{min_area_px, simplify_fraction, regularize, …}`.

`config/site.example.yaml` is the annotated template — copy it to
`config/site.yaml` and fill in what you know.

---

## 10. How to run (end-to-end workflow)

```bash
# 0. install (once)
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# 1. put an image in input/
cp my-image.jpg input/site.jpg

# 2. copy the config template and fill in scale (+ sun, if you want heights)
cp config/site.example.yaml config/site.yaml
#   scale: two pixel points a known distance apart is the most reliable option.
#   sun:   elevation_deg directly, OR geo.center_lat/lon + sun.timestamp.

# 3. analyze
satrecon analyze input/site.jpg -c config/site.yaml
#   prints buildings count, scale known?, heights N/N from shadows

# 4. REVIEW (required, not optional): open output/site/review.html
#   check footprints against the image; tune detector.params.accept_score and
#   re-run if detections are poor. Confirm heights look sane.

# 5. generate the 3D model
satrecon generate data/site.json --formats glb,obj
#   -> output/site/model/site.glb  (drop into any glTF viewer)
```

Heights require **both** a scale and a sun elevation. Without them, footprints
are still produced and `generate` extrudes them at the placeholder height so the
layout is viewable; the export units (`pixels` vs `meters`) and the scene notes
say which case you are in.

---

## 11. Testing

`pytest -q` (73 tests at time of writing). What they cover:

- `test_scale.py` — scale never invents metres; resolution order.
- `test_sun.py` — NOAA astronomy (overhead at equator/equinox, noon symmetry,
  solstice elevation, east→west azimuth, timezone conversion), unknown handling,
  the image-frame shadow-direction cross-check.
- `test_height.py` — pure geometry (`height = shadow_len × tan`), floors, and the
  image measurement against **synthetic scenes with a known shadow length**.
- `test_geometry.py` — ear-clipping correctness (area conservation, concave L),
  vertex/face counts, watertightness (every edge shared by exactly two
  triangles), scale→metres, and GLB container validity (parsed back from bytes).
- `test_pipeline_phase2.py` — full image→…→height→GLB on a synthetic scene,
  recovering a known ground-truth height; reproducibility including height.
- `test_model.py`, `test_footprint.py`, `test_pipeline.py` — model round-trips,
  confidence bands, footprint cleanup, Phase 1 wiring.

**Testing philosophy:** validate against inputs whose answer is known
independently (hand-computed physics, synthetic images with a drawn shadow of a
known length, open ground-truth datasets). Real imagery has no known answer to
check against, so it is for exploration, not for asserting correctness.

---

## 12. Design rules (invariants to preserve)

1. **Never invent a scale or a sun elevation.** No image-only fallback for
   either. Unknown ⇒ the dependent quantity stays `null`/pixels.
2. **Every inferred value carries a `source` and a `confidence`.**
3. **`null` ("not estimated") is not `0.0` ("estimated, unreliable").**
4. **Source data (scene JSON) is separate from generated geometry (mesh).**
5. **User edits (`user_edited`) are never silently overwritten.**
6. **Determinism** — verifiable via `resultHash` and `satrecon verify`.
7. **No heavy dependencies** — NumPy/OpenCV/Pillow/PyYAML only; exporters are
   hand-written. Keep new core features dependency-light; put anything heavy
   (ML runtimes, geospatial libs) behind an optional extra and an adapter.

---

## 13. Open work & how to extend (the roadmap)

The system detects footprints classically today. The classical detector is the
weakest link on complex real imagery; it is *designed* to be replaced. Priority
items, each a clean plug-in that reuses the whole existing height/3D chain:

### 13.1 A "manual/imported footprints" detector  *(fastest, high value)*

Add `detect/manual.py`: a `BuildingDetector` named `manual` that reads polygons
from a **GeoJSON (or simple JSON) file** given in `detector.params.path`,
converts each ring to a `BuildingDetection` (in working-image pixels), and skips
detection entirely. This lets a user (or another tool) supply exact footprints
and still get the full shadow-based height estimation, 3D extrusion and export.
Register it and select via `detector: { name: manual, params: { path: ... } }`.
Watch the coordinate space: footprints must be in working-image pixels, so if
the GeoJSON is in source pixels, divide by `LoadedImage.scale_factor`.

### 13.2 A learned/vision segmentation detector

Add `detect/segmentation_model.py` wrapping a building-segmentation model (e.g.
an ONNX/`onnxruntime` model, or any vision interface) that outputs a building
mask; convert connected components to contours → `BuildingDetection[]`. Keep the
runtime behind an optional dependency and the model path in `params`. This is
the main accuracy upgrade over the classical detector for dense/complex scenes.

### 13.3 Accuracy validation against open ground truth

Build a small evaluation harness that pulls **OpenStreetMap** building
footprints (and `building:levels`) for a geolocated open test area, aligns them
to the scene, and reports precision/recall on footprints and error distributions
on heights/floors. This is what turns "looks about right" into a measured number
and lets any detector be compared objectively. Requires real-world coordinates
(`geo` section) and is the right place to integrate `rasterio`/GDAL for
georeferenced GeoTIFF inputs (behind an adapter).

### 13.4 Roof-type inference and stepped heights  *(Phase 3)*

Today roofs are assumed flat and one height is applied per footprint. Next:
infer flat/pitched/complex from intra-roof shading and shadow structure, and
support multiple heights per footprint (a building with wings of different
heights) by measuring shadow length per edge rather than once per building.

### 13.5 3D viewer + correction workflow  *(Phase 4)*

The 2D `review.html` is the seed of a 3D viewer (same scene JSON, same
confidence bands). Add orbit/pan/zoom, the source image as a scaled ground
plane, and interactive corrections (move vertices, change a height, delete a
false detection, add a missing building) that write back to the scene JSON and
set `user_edited: true`.

### How to add any new stage or detector (checklist)

1. Put a new **stage** in `stages/<name>.py` as a pure function
   `run(inputs, params) -> result`; wire it in `pipeline.py`; add its defaults to
   `config.py::DEFAULTS`; write tests against known-answer inputs.
2. Put a new **detector** in `detect/<name>.py` implementing `BuildingDetector`;
   `register_detector("<name>", Factory)`; select via config. Return polygons in
   working-image pixels.
3. If output numbers change for identical input, bump `MODEL_VERSION` in
   `__init__.py` (it is recorded in every scene for cross-version comparison).
4. Keep new heavy dependencies optional (extra + adapter), preserving the
   "runs locally with no GPU/network" property.

---

## 14. Quick facts

- **Language / runtime:** Python ≥ 3.10.
- **Core deps:** numpy, opencv-python-headless, Pillow, PyYAML. **Dev:** pytest.
- **Entry point:** `satrecon` (`satrecon.cli:main`).
- **Coordinate spaces:** *working* pixels (downscaled, used internally) vs
  *source* pixels (persisted); `LoadedImage.scale_factor` maps between them. 3D
  world is Y-up, metres when scaled.
- **Reproducibility:** `resultHash` over geometry + heights (not time);
  `satrecon verify` re-runs and compares.
- **Where to look first:** `pipeline.py` (the wiring), `model.py` (the schema),
  `detect/base.py` (the extension point), `docs/architecture.md` (the *why*).
