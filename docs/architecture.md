# Architecture and design decisions

This document records *why* the pipeline is shaped the way it is, and what the
Phase 2+ stages will need. It is the companion to the stage table in the README.

## Design rules

1. **Never invent a scale.** A wrong metres-per-pixel silently corrupts every
   dimension, height and floor count downstream, and the error is invisible
   because the numbers still look plausible. If the user has not supplied a
   scale, dimensions stay in pixels. There is no image-only fallback.
2. **Every inferred value carries a source and a confidence.** `height_source`,
   `floors_source`, `roof_source` and `scale.source` are part of the schema, not
   an afterthought.
3. **"Not estimated" is not "zero confidence."** `null` is excluded from the
   overall confidence; `0.0` is included. Conflating them would make an unknown
   scale look like a bad footprint.
4. **Source data stays separate from generated geometry.** The scene file
   records observations. 3D meshes are derived artifacts and are regenerated,
   never edited in place.
5. **User edits are not silently overwritten.** `Building.user_edited` exists
   from Phase 1 so the Phase 4 correction workflow has somewhere to record that
   a record is authoritative.
6. **Determinism.** No unseeded randomness anywhere. The one place a random
   generator appears — debug label colouring — is explicitly seeded.

## Why the detector is region-based

The first approaches tried and rejected on real imagery:

| Approach | Why it failed |
|---|---|
| Global brightness + texture threshold | Roads, parking and concrete are as bright and as textured as roofs. The whole site merged into one 800k-pixel component. |
| Opening by reconstruction to remove thin ribbons | The "built surface" mask covered 77% of the image and was fully connected, so reconstruction flooded everything. |
| Superpixels (SLIC) + per-superpixel scoring | Boundaries cut across roof edges; only the brightest sunlit facets survived, producing fragments rather than buildings. |

What worked: a gradient watershed whose seeds are the low-gradient plateaus of
the mean-shift-smoothed image. Region borders then follow real roof edges, and
classification can use *shape* — which is what actually separates a roof from a
road.

The discriminating features, in rough order of usefulness:

- **elongation** (min-area-rect long/short) — roads and kerbs are ribbons;
- **thickness** (max inscribed radius) — a region narrower than the surrounding
  paths is not a footprint;
- **shadow adjacency** — in a nadir image, an attached shadow on a consistent
  side is the strongest evidence a bright blob is raised;
- **rectangularity and solidity** — built structure versus organic shapes;
- **brightness**, with data-adaptive anchors (L percentiles 50 and 93) so pale
  concrete and dark roof scenes both get a usable ramp;
- **vegetation / shadow / bare-ground fractions** as penalties.

Scoring is a weighted *mean* of the positive cues minus penalties, so the accept
threshold keeps its meaning when weights are retuned.

Large roofs break into several watershed facets, so accepted regions that touch
and agree on lightness are merged into one building instance; facets that
disagree strongly are split back out.

## Shadow direction

Estimated by finding the in-plane offset that best aligns the shadow mask with
bright surfaces, scored over all angles at several distances. The confidence is
the peakedness of the response, so a scene with few or ambiguous shadows
attenuates the shadow cue rather than trusting a noisy direction.

This stage exists on its own — rather than inside the detector — because
Phase 2 needs exactly the same direction for shadow-based height estimation.

## Phase 2: height and floors (not yet implemented)

Planned strategies, in priority order, each recording its own source:

1. `user_supplied` — a height given in the config or via correction.
2. `shadow` — for a building with a measurable attached shadow,
   `height ≈ shadow_length × tan(solar_elevation)`. This needs the shadow run
   length along the estimated direction *and* a solar elevation. Elevation can
   come from the config, or from a capture timestamp plus coordinates. Without
   one, only *relative* heights are recoverable, and they must be labelled as
   such. Uncertainty is dominated by the elevation estimate and by shadows
   falling on sloped or occluded ground.
3. `facade` — where an off-nadir view shows a facade, its lean gives height.
4. `heuristic` — a generic fallback, with low confidence, clearly marked.

Floors follow from `round(height / floor_height)` with per-use-type floor
heights from the config. The building use type must not be guessed: when there
is no evidence, the assumption used is recorded and the confidence lowered.

## Phase 3: 3D generation

Recommended split, to be confirmed before implementation:

- **glTF/GLB as the interchange format**, written directly. It carries per-mesh
  metadata via `extras`, loads in every web viewer, and imports into Blender
  without a Blender dependency in the core pipeline.
- **Blender as an optional target**, driven by a standalone `bpy` script that
  reads the same scene JSON. Keeping Blender out of the core means the pipeline
  runs without it installed, which matters for reproducibility and CI.
- Geometry stays procedural: the mesh is a pure function of the scene JSON, so
  correcting a footprint and regenerating is always safe.

## Phase 4: viewer and corrections

The Phase 1 review page is the seed of the 3D viewer: the same scene JSON, the
same confidence bands, the same selection panel. The 3D viewer adds orbit/pan/
zoom, a top-down mode, the source image as a scaled ground plane, and layer
toggles.

Corrections write back to the scene JSON and set `user_edited: true`. The
pipeline must then treat those buildings as fixed and regenerate everything
else around them.

## External data sources (evaluated, not yet integrated)

Adapters are the intended shape — `TerrainProvider`, `BuildingDataProvider`,
`ImageAnalysisProvider`, `ThreeDExporter` — so no provider is load-bearing.

- **OpenStreetMap** — where coverage exists, published building footprints and
  `building:levels` tags are far better than anything inferable from one image,
  and could be used to validate detections. Requires real-world coordinates,
  which the pipeline does not assume it has.
- **Open elevation datasets (SRTM, Copernicus DEM)** — terrain, not building
  height; useful for a ground plane, not for structure heights.
- **GDAL / rasterio** — only needed for georeferenced inputs (GeoTIFF). Worth
  adding when a user supplies one, not before.

All of these require the image to be geolocated. Since this pipeline is designed
to work on an image with no metadata at all, they stay optional adapters behind
an interface.
