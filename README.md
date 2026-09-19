# satrecon

A local pipeline that turns a satellite or aerial image you supply into an
approximate, editable 3D representation of the built environment visible in it.

**Current status: Phase 2 — image → detection → footprints → heights → 3D model.**
Building heights and floor counts are estimated from cast shadows, and reviewed
footprints are extruded into a 3D mesh you can open in Blender, three.js or any
glTF viewer (GLB / glTF / OBJ). Heights are only produced when a scale *and* a
sun elevation are available; without them the pipeline stops at footprints, and
every height stays `null` rather than being guessed. Roof shape is still assumed
flat.

## What this is and is not

Everything produced here is an **approximation inferred from pixels**. It is
not a measurement of any real structure, and no value should be treated as
survey data. Specifically:

- Footprints come from image segmentation and will contain false positives
  (roads, kerbs, tree lines) and misses. Reviewing them is a required step,
  not an optional one.
- Dimensions are in **pixels** unless you supply a scale. The pipeline never
  invents one.
- Nothing is inferred about what any structure is used for. The detector only
  distinguishes "roof-like surface" from vegetation, shadow and bare ground.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Requires Python 3.10+. Phase 1 needs only NumPy, OpenCV, Pillow and PyYAML —
no GPU, no network access, no external service.

## Use

```bash
# 1. Put an image in input/
cp my-image.jpg input/site.jpg

# 2. Detect buildings and extract footprints
satrecon analyze input/site.jpg --config config/site.example.yaml

# 3. Open the review page and check the footprints against the image
#    output/site/review.html

# 4. Build and export the 3D model (GLB + OBJ by default)
satrecon generate data/site.json

# 5. Confirm a run reproduces exactly
satrecon verify data/site.json
```

Heights need a **scale** and a **sun elevation** (see "Telling the pipeline what
you know"). With those set, `analyze` reports `heights N/N from shadows` and
`generate` extrudes each footprint to its measured height. Without them,
footprints are still produced and `generate` extrudes them at a placeholder
height so the layout is still viewable — the export's units and the scene notes
say which case you are in.

Without installing, prefix with `PYTHONPATH=src python3 -m satrecon`.

### Output

| Path | What it is |
|---|---|
| `data/<name>.json` | The scene: footprints, heights, floors, confidences, provenance |
| `output/<name>/overlay.png` | Footprints drawn over the source image |
| `output/<name>/footprints.png` | Geometry alone, on a neutral background |
| `output/<name>/review.html` | Interactive 2D review page |
| `output/<name>/stages/*.png` | Every intermediate raster the detector used |
| `output/<name>/model/<name>.glb` | 3D model (binary glTF); also `.obj`, `.gltf` on request |

The stage rasters exist so that a wrong footprint can be traced to the cue that
caused it, rather than being a black box.

## Telling the pipeline what you know

Copy `config/site.example.yaml` and fill in whatever you actually know. Values
you supply are authoritative and are never overwritten by a heuristic.

The most useful thing you can provide is a **scale**. Either set
`scale.meters_per_pixel` directly, or point at two pixels a known distance
apart:

```yaml
scale:
  reference:
    point_a: [412, 980]
    point_b: [712, 980]
    distance_meters: 75
```

With no scale, every dimension stays in pixels and the review page says
`Scale: UNKNOWN` in the scene panel.

The second thing height estimation needs is the **sun elevation**. State it
directly, or give coordinates and a capture time and let it be computed:

```yaml
sun:
  elevation_deg: 58            # option A: state it
  # or, option B — computed from geo + a timestamp with a UTC offset:
geo:
  center_lat: 32.081
  center_lon: 34.780
sun:
  timestamp: 2024-06-01T10:30:00+03:00
```

The height for each building is `shadow_length × tan(elevation)`, measured by
casting rays from the shadow-facing edge of the footprint along the shadow
direction. It is an estimate with a confidence, not a survey; a flat roof is
assumed. When the sun's azimuth is known (computed case) the scene notes include
a cross-check between the shadow direction recovered from the image and the one
the sun implies — a large disagreement is a warning that the heights are
unreliable.

## Architecture

Each arrow is a module boundary. Stages do not reach across each other.

```
 image ─▶ preprocess ─▶ surface cues ─▶ shadow direction ─▶ segmentation
                                                                 │
                                                                 ▼
   footprints ◀─ detection (pluggable) ◀──────────────────────────┘
        │
        ▼
 solar geometry ─▶ height (shadow rays) ─▶ 3D extrude ─▶ GLB / OBJ / glTF
```

| Stage | Module | Job |
|---|---|---|
| Load | `imaging.py` | Decode, hash, pick a working resolution |
| Preprocess | `stages/preprocess.py` | Edge-preserving mean-shift smoothing |
| Cues | `stages/cues.py` | Vegetation / shadow / bare-ground / texture masks |
| Shadow | `stages/shadow.py` | Dominant illumination direction |
| Segment | `stages/segmentation.py` | Gradient watershed into regions |
| Features | `stages/regionfeatures.py` | Per-region shape and radiometry |
| Detect | `detect/classical.py` | Score regions, merge roof facets |
| Footprints | `stages/footprint.py` | Simplify, regularise, map to source pixels |
| Scale | `stages/scale.py` | Resolve m/px from user input only |
| Sun | `stages/sun.py` | Solar elevation/azimuth from geo+time, or stated |
| Height | `stages/height.py` | Shadow-ray height and floor estimation |
| Extrude | `geometry/mesh.py` | Footprint + height → watertight prism |
| Export | `geometry/export.py` | GLB / glTF / OBJ writers (no dependencies) |
| Review | `debug/` | Overlays, stage rasters, interactive page |

### Why region-based detection

Global colour thresholds cannot separate roofs from roads — both are bright,
low-chroma surfaces, and thresholding merges the whole site into one blob. What
separates them is the *boundary*: roofs are bounded by strong closed edges. So
the image is partitioned into edge-following regions first, and classification
happens per region, using shape (thickness, elongation, rectangularity,
solidity) alongside colour.

### Swapping the detector

`detect/base.py` defines the contract:

```python
detector.detect(context) -> DetectorResult   # BuildingDetection[]
```

A detector receives the already-computed stages and returns polygons in working
image coordinates. To plug in a segmentation model, a remote vision API or a
file of hand-drawn polygons, implement `BuildingDetector`, call
`register_detector("name", Factory)`, and set `detector.name` in the config.
Nothing downstream knows which detector ran.

## Confidence

Every building carries per-aspect confidence. `null` means *not estimated in
this run* and is excluded from the average — deliberately different from `0.0`,
which means *estimated, and unreliable*. `height` and `floors` are `null` when
no shadow-based height could be measured (no scale, no sun, or no visible
shadow), and `scale` is `null` when no scale was supplied. A height confidence
combines how much the shadow rays agreed with how trustworthy the shadow
direction and sun position were — a good measurement in a wrongly-estimated
direction is still wrong, so the weakest input caps the result.

Bands shown in the review page and overlay: **HIGH** ≥ 0.66, **MEDIUM** ≥ 0.40,
otherwise **LOW**.

## Reproducibility

The same image plus the same config produces the same result. Every scene
records the image SHA-256, the effective config fingerprint, the detector, the
model version and a `resultHash` computed over the geometry but *not* the
timestamp. `satrecon verify` re-runs the analysis and compares.

## Accuracy and limitations

The classical detector assumes:

- a near-nadir view, so a roof projects roughly onto its own footprint;
- roofs brighter and less chromatic than vegetation and bare ground;
- one dominant illumination direction across the image;
- buildings wider than the paths and roads around them.

Where those do not hold, expect misses and false positives. Known weaknesses on
real imagery: dark roofs, roofs the same tone as adjacent paving, buildings
under tree canopy, tall structures leaning off their footprint away from nadir,
long narrow buildings suppressed by the elongation penalty, and wide plazas or
parking areas accepted as roofs. Default parameters were tuned against a single
image; expect to adjust `detector.params.accept_score` per image.

Height estimation adds its own assumptions: a flat roof, flat ground around the
building, and a shadow that falls on open ground rather than onto a neighbour.
Shadows that are occluded, merge with an adjacent building's shadow, or fall on
a slope will be mismeasured. The height is `shadow_length × tan(elevation)`, so
error grows quickly at low sun elevations. Treat every height as an estimate
with the confidence attached, and use the shadow-direction cross-check (when the
sun azimuth is known) as a sanity gate.

## Roadmap

- **Phase 1 (done)** — detection, footprints, 2D review, reproducibility.
- **Phase 2 (done)** — scale-aware dimensions, shadow-based height estimation
  with explicit uncertainty, floor-count estimation from configurable floor
  heights, procedural extrusion and GLB/glTF/OBJ export with units preserved.
- **Phase 3** — roof-type inference where evidence supports it, per-building
  height (stepped roofs) rather than one height per footprint, optional Blender
  scene assembly.
- **Phase 4** — 3D viewer with the source image as a scaled ground plane, and
  the manual correction workflow (move vertices, change height, delete a false
  detection, add a missing building) writing back to the scene file.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```
