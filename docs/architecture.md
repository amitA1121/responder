# Architecture and design decisions

This document records *why* the pipeline is shaped the way it is. Phases 1 and 2
are implemented; Phases 3–4 are still planned. It is the companion to the stage
table in the README.

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
height estimation needs exactly the same direction.

## Phase 2: solar geometry, height and floors

### Solar elevation is resolved like scale

`stages/sun.py` mirrors `stages/scale.py` exactly: a value is used only if it
can be traced to something the user supplied, and there is no image-only
fallback. Resolution order is (1) an explicit `sun.elevation_deg`, (2) the NOAA
solar-position algorithm applied to `geo.center_lat/lon` plus a timestamp with a
UTC offset, else (3) unknown, in which case heights are not estimated. A
computed position below the horizon is reported unknown rather than negative —
that means the inputs are wrong, not that the building has negative height.

The computed case also yields the sun's azimuth, which gives the shadow
direction the sun *implies*. That is compared against the direction recovered
from the image and surfaced in the scene notes as a cross-check; the image
direction is never silently overridden by it.

### Height from shadows

`stages/height.py` measures a shadow length and converts it:
`height = shadow_length × tan(elevation)`. The measurement casts rays outward
from the shadow-facing footprint edges (selected by the sign of each edge's
outward normal against the shadow direction) and counts how far the shadow cue
persists, tolerating small gaps. The robust aggregate of the ray lengths is the
shadow length; the spread of the rays becomes a per-building agreement score.

The pure geometry (`height_from_shadow_m`, `estimated_floors`) is separated from
the image measurement so it can be tested against hand-computed values, and the
image measurement is tested against synthetic scenes with a known shadow length.

Confidence is multiplicative across the weakest inputs — ray agreement × shadow-
direction confidence × sun confidence, capped further when few rays contributed
— because a precise measurement in a wrongly-estimated direction is still wrong.
Floors follow from `round(height / floor_height)` with a configurable floor
height, floored at one so a real building is never zero storeys.

Not implemented, deferred to Phase 3: facade-based height from off-nadir views,
per-building stepped heights, and any inference of building *use* (still never
guessed).

## Phase 3: 3D generation (extrusion done; roofs and Blender pending)

Implemented in `geometry/`:

- **Extrusion** (`mesh.py`): each reviewed footprint is lifted into a watertight
  prism — floor cap, roof cap and wall band. Footprints are frequently concave
  (L/U/cross), so the caps are triangulated by **ear clipping** rather than
  assuming convexity; the result is verified watertight (every edge shared by
  exactly two triangles) in tests. Coordinates are metres when a scale is known
  and pixels otherwise, and the unit travels with the export.
- **Export** (`export.py`): GLB, text glTF and OBJ are written by hand with no
  third-party dependency, keeping Phase 1's "NumPy/OpenCV only" property. GLB
  carries the unit in `asset.extras`; the files load in three.js, Blender and
  standard glTF viewers.
- Geometry stays procedural: the mesh is a pure function of the scene JSON, so
  correcting a footprint and regenerating is always safe.

Still planned for Phase 3: roof-type inference where evidence supports it (roofs
are assumed flat today) and an optional Blender target driven by a standalone
`bpy` script reading the same scene JSON, so the core never depends on Blender.

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
