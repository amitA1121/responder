"""Extrude reviewed footprints into 3D prisms.

A footprint is a flat polygon in image pixels; a height is a number of metres.
This module lifts the two into a watertight prism: a floor cap, a roof cap and
a wall band between them. Footprints are often concave (L-, U-, cross-shaped),
so the caps are triangulated by ear clipping rather than assuming convexity.

Coordinates follow the glTF convention used by the exporters: X east, Z south
(image +y), Y up. One world unit is one metre when a scale is known; when it is
not, geometry is built in pixels and the exporter records that the units are
pixels, so a viewer never silently mislabels them as metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logging_setup import get_logger
from ..model import Building, ScaleInfo

log = get_logger(__name__)


@dataclass
class Mesh:
    """A triangle mesh. `vertices` is (N, 3) float; `faces` is (M, 3) int."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    # Parallel to `faces`: the building id each triangle belongs to, so a
    # viewer can select or colour a single structure.
    face_building: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0

    def merged_with(self, other: "Mesh") -> "Mesh":
        offset = len(self.vertices)
        vertices = np.vstack([self.vertices, other.vertices]) if len(other.vertices) else self.vertices
        faces = (
            np.vstack([self.faces, other.faces + offset])
            if len(other.faces)
            else self.faces
        )
        return Mesh(vertices, faces, self.face_building + other.face_building)


def _signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def _dedupe_closed(polygon: np.ndarray) -> np.ndarray:
    """Drop a repeated closing vertex and any zero-length edges."""
    keep: list[np.ndarray] = []
    count = len(polygon)
    for i in range(count):
        current = polygon[i]
        nxt = polygon[(i + 1) % count]
        if np.hypot(*(nxt - current)) > 1e-9:
            keep.append(current)
    return np.array(keep, dtype=np.float64) if keep else polygon


def _point_in_triangle(p, a, b, c) -> bool:
    # Barycentric sign test.
    v0, v1, v2 = c - a, b - a, p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9


def triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon (concave allowed).

    Returns index triples into the *input* polygon order. Robust enough for
    building footprints; it is not a general mesher and will bail out (empty)
    on self-intersecting rings rather than emit garbage triangles.
    """
    poly = _dedupe_closed(np.asarray(polygon, dtype=np.float64))
    n = len(poly)
    if n < 3:
        return []
    # Work anticlockwise so "convex vertex" has a consistent sign.
    indices = list(range(n))
    if _signed_area(poly) < 0:
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    max_guard = 2 * n * n + 16
    while len(indices) > 3 and guard < max_guard:
        guard += 1
        ear_found = False
        m = len(indices)
        for k in range(m):
            i_prev, i_curr, i_next = indices[k - 1], indices[k], indices[(k + 1) % m]
            a, b, c = poly[i_prev], poly[i_curr], poly[i_next]
            # Convex corner? Cross product > 0 in a CCW ring.
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-9:
                continue
            # No other vertex may lie inside the candidate ear.
            contains = False
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle(poly[j], a, b, c):
                    contains = True
                    break
            if contains:
                continue
            triangles.append((i_prev, i_curr, i_next))
            indices.pop(k)
            ear_found = True
            break
        if not ear_found:
            log.warning("ear clipping stalled (%d vertices left); footprint may self-intersect", len(indices))
            return []
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    # Indices already refer to `poly`, which is the caller's (deduped) order;
    # reversing `indices` above only changed traversal, not the numbering.
    return triangles


def building_to_mesh(building: Building, scale: ScaleInfo, default_height_units: float) -> Mesh:
    """Extrude one building. Height falls back to `default_height_units` (in the
    working unit - metres if scaled, else pixels) when none was estimated, so a
    reviewed-but-unmeasured footprint still appears as a low block.
    """
    footprint = _dedupe_closed(np.asarray(building.footprint, dtype=np.float64))
    n = len(footprint)
    if n < 3:
        return Mesh()

    unit = scale.meters_per_pixel if scale.known else 1.0
    # A reviewed footprint with no measured height still gets a placeholder
    # block so it is visible; the height source in the data model records which
    # buildings are real measurements versus this fallback.
    if building.estimated_height_m is not None:
        height = float(building.estimated_height_m)
    else:
        height = float(default_height_units)
    height = max(height, 1e-3)

    # Ground plane: image (x, y) px -> world (X east, Z south) in units.
    ground = footprint * unit
    base = np.column_stack([ground[:, 0], np.zeros(n), ground[:, 1]])      # y = 0
    top = np.column_stack([ground[:, 0], np.full(n, height), ground[:, 1]])  # y = height
    vertices = np.vstack([base, top])

    faces: list[tuple[int, int, int]] = []

    # Roof cap (top ring), triangulated, facing up.
    cap = triangulate(footprint)
    for (i, j, k) in cap:
        faces.append((n + i, n + j, n + k))
    # Floor cap, reversed winding so it faces down.
    for (i, j, k) in cap:
        faces.append((i, k, j))

    # Walls: quad per edge, split into two triangles.
    for i in range(n):
        j = (i + 1) % n
        b0, b1 = i, j
        t0, t1 = n + i, n + j
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))

    face_ids = [building.id] * len(faces)
    return Mesh(vertices, np.array(faces, dtype=np.int64), face_ids)


def scene_to_mesh(
    buildings: list[Building], scale: ScaleInfo, default_height_units: float = 3.0
) -> Mesh:
    """Extrude every building into one combined mesh."""
    combined = Mesh()
    built = 0
    for building in buildings:
        mesh = building_to_mesh(building, scale, default_height_units)
        if not mesh.is_empty:
            combined = combined.merged_with(mesh)
            built += 1
    log.info("built %d building meshes (%d vertices, %d triangles)",
             built, len(combined.vertices), len(combined.faces))
    return combined
