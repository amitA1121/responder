# satrecon

A local pipeline that turns a satellite or aerial image you supply into an
approximate, editable 3D representation of the built environment visible in it.

**Current status: Phase 1 — image → detection → footprints → 2D review.**
Heights, floor counts, roof types and 3D geometry are not generated yet. The
data model already carries those fields as `null` so the format will not change
underneath the viewer when Phase 2 lands.

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

# 4. Confirm a run reproduces exactly
satrecon verify data/site.json
```

Without installing, prefix with `PYTHONPATH=src python3 -m satrecon`.

### Output

| Path | What it is |
|---|---|
| `data/<name>.json` | The scene: footprints, confidences, provenance |
| `output/<name>/overlay.png` | Footprints drawn over the source image |
| `output/<name>/footprints.png` | Geometry alone, on a neutral background |
| `output/<name>/review.html` | Interactive 2D review page |
| `output/<name>/stages/*.png` | Every intermediate raster the detector used |

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

## Architecture

Each arrow is a module boundary. Stages do not reach across each other.

```
 image ─▶ preprocess ─▶ surface cues ─▶ shadow direction ─▶ segmentation
                                                                 │
                                                                 ▼
 2D review ◀─ footprints ◀─ detection (pluggable) ◀───────────────┘
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
which means *estimated, and unreliable*. In Phase 1, `height` and `floors` are
always `null`, and `scale` is `null` when no scale was supplied.

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

## Roadmap

- **Phase 1 (done)** — detection, footprints, 2D review, reproducibility.
- **Phase 2** — scale-aware dimensions, shadow-based height estimation with
  explicit uncertainty, floor-count estimation from configurable floor heights,
  roof-type inference where it can be supported by evidence.
- **Phase 3** — procedural extrusion, scene assembly, GLB/glTF export with
  metadata preserved, optional Blender scene generation.
- **Phase 4** — 3D viewer with the source image as a scaled ground plane, and
  the manual correction workflow (move vertices, change height, delete a false
  detection, add a missing building) writing back to the scene file.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```
