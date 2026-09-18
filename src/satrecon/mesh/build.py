"""Turn scene geometry into triangle meshes.

Coordinates here stay in scene space: X/Y on the ground, Z up, metres. The
exporter converts to the target convention, so this module never has to care
which engine is downstream.

Flat shading throughout - vertices are duplicated per face rather than shared.
That is the correct look for architecture (crisp edges, no smeared corners) and
it keeps normal handling trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .triangulate import ensure_ccw, triangulate


@dataclass
class BuildingMesh:
    positions: np.ndarray      # (N, 3) float32
    normals: np.ndarray        # (N, 3) float32
    indices: np.ndarray        # (M,) uint32
    material: str = "default"
    extras: dict = field(default_factory=dict)


class _Accumulator:
    """Collects flat-shaded faces and emits arrays."""

    def __init__(self) -> None:
        self.positions: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.indices: list[int] = []

    def add_triangle(self, a, b, c) -> None:
        a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            return           # degenerate face, skip
        normal = normal / length
        base = len(self.positions)
        self.positions.extend([a, b, c])
        self.normals.extend([normal, normal, normal])
        self.indices.extend([base, base + 1, base + 2])

    def add_quad(self, a, b, c, d) -> None:
        self.add_triangle(a, b, c)
        self.add_triangle(a, c, d)

    def finish(self, material: str, extras: dict) -> BuildingMesh:
        if not self.positions:
            empty = np.zeros((0, 3), np.float32)
            return BuildingMesh(empty, empty, np.zeros((0,), np.uint32), material, extras)
        return BuildingMesh(
            positions=np.asarray(self.positions, dtype=np.float32),
            normals=np.asarray(self.normals, dtype=np.float32),
            indices=np.asarray(self.indices, dtype=np.uint32),
            material=material,
            extras=extras,
        )


def _walls(acc: _Accumulator, polygon: np.ndarray, base_z: float, top_z: float) -> None:
    count = len(polygon)
    for i in range(count):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % count]
        # CCW footprint viewed from above -> this winding faces outward.
        acc.add_quad(
            (p0[0], p0[1], base_z),
            (p1[0], p1[1], base_z),
            (p1[0], p1[1], top_z),
            (p0[0], p0[1], top_z),
        )


def _flat_roof(acc: _Accumulator, polygon: np.ndarray, z: float) -> None:
    for a, b, c in triangulate(polygon):
        acc.add_triangle(
            (polygon[a][0], polygon[a][1], z),
            (polygon[b][0], polygon[b][1], z),
            (polygon[c][0], polygon[c][1], z),
        )


def _long_axis_ends(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """For a 4-gon, return the two ridge-line endpoints and the corner order.

    The ridge runs parallel to the longer pair of edges, along the midline.
    """
    p0, p1, p2, p3 = polygon
    if np.linalg.norm(p1 - p0) >= np.linalg.norm(p2 - p1):
        order = [0, 1, 2, 3]
    else:
        order = [1, 2, 3, 0]
    q0, q1, q2, q3 = (polygon[i] for i in order)
    return (q0 + q3) / 2.0, (q1 + q2) / 2.0, order


def _pitched_roof(
    acc: _Accumulator, polygon: np.ndarray, eave_z: float, rise: float, hip_inset: float
) -> None:
    """Gabled roof when `hip_inset` is 0, hipped when it is positive."""
    m0, m1, order = _long_axis_ends(polygon)
    q0, q1, q2, q3 = (polygon[i] for i in order)

    ridge = m1 - m0
    length = float(np.linalg.norm(ridge))
    if length < 1e-9:
        _flat_roof(acc, polygon, eave_z)
        return
    direction = ridge / length
    inset = min(hip_inset, length * 0.45)
    r0 = m0 + direction * inset
    r1 = m1 - direction * inset
    ridge_z = eave_z + rise

    a0 = (q0[0], q0[1], eave_z)
    a1 = (q1[0], q1[1], eave_z)
    a2 = (q2[0], q2[1], eave_z)
    a3 = (q3[0], q3[1], eave_z)
    t0 = (r0[0], r0[1], ridge_z)
    t1 = (r1[0], r1[1], ridge_z)

    # The two long slopes.
    acc.add_quad(a0, a1, t1, t0)
    acc.add_quad(a2, a3, t0, t1)

    if inset > 1e-6:
        # Hipped: the short ends are triangular slopes.
        acc.add_triangle(a1, a2, t1)
        acc.add_triangle(a3, a0, t0)
    else:
        # Gabled: the short ends are vertical walls up to the ridge.
        acc.add_triangle(a1, a2, t1)
        acc.add_triangle(a3, a0, t0)


def build_building(
    footprint: list[tuple[float, float]],
    height_m: float,
    roof_type: str = "flat",
    roof_pitch_m: float = 2.6,
    material: str = "default",
    extras: dict | None = None,
) -> BuildingMesh:
    """Extrude a footprint to `height_m` and cap it with the requested roof."""
    polygon = ensure_ccw(np.asarray(footprint, dtype=np.float64))
    if len(polygon) < 3 or height_m <= 0:
        return _Accumulator().finish(material, extras or {})

    acc = _Accumulator()
    _walls(acc, polygon, 0.0, height_m)

    # Pitched roofs are only well defined here for quadrilateral footprints;
    # anything more complex is capped flat and the caller is told via extras.
    resolved = roof_type
    if roof_type in ("gabled", "hip") and len(polygon) == 4:
        _pitched_roof(
            acc, polygon, height_m, roof_pitch_m, 0.0 if roof_type == "gabled" else roof_pitch_m * 1.4
        )
    else:
        if roof_type in ("gabled", "hip"):
            resolved = "flat"
        _flat_roof(acc, polygon, height_m)

    merged = dict(extras or {})
    merged["roofTypeRequested"] = roof_type
    merged["roofTypeBuilt"] = resolved
    return acc.finish(material, merged)


def build_ground(width: float, depth: float, z: float = 0.0) -> BuildingMesh:
    acc = _Accumulator()
    acc.add_quad((0, 0, z), (width, 0, z), (width, depth, z), (0, depth, z))
    return acc.finish("ground", {"kind": "ground"})


def build_roads(polygons: list[list[tuple[float, float]]], z: float = 0.02) -> BuildingMesh:
    """All road surfaces as one mesh - they are never edited individually."""
    acc = _Accumulator()
    for polygon in polygons:
        ring = ensure_ccw(np.asarray(polygon, dtype=np.float64))
        for a, b, c in triangulate(ring):
            acc.add_triangle(
                (ring[a][0], ring[a][1], z),
                (ring[b][0], ring[b][1], z),
                (ring[c][0], ring[c][1], z),
            )
    return acc.finish("road", {"kind": "roads"})
