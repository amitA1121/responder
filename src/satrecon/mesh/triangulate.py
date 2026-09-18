"""Ear-clipping triangulation for simple polygons.

Written out rather than pulled from a dependency because it is the only
computational geometry the mesher needs, and keeping the dependency list short
matters more here than saving sixty lines.

Handles concave polygons (L-shapes, U-shapes) but not holes or
self-intersection; the caller is responsible for supplying a simple polygon.
"""

from __future__ import annotations

import numpy as np


def signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    return polygon if signed_area(polygon) > 0 else polygon[::-1].copy()


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    has_negative = min(d1, d2, d3) < -1e-12
    has_positive = max(d1, d2, d3) > 1e-12
    return not (has_negative and has_positive)


def triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Return triangles as index triples into the ORIGINAL polygon array."""
    count = len(polygon)
    if count < 3:
        return []
    if count == 3:
        return [(0, 1, 2)]

    ccw = signed_area(polygon) > 0
    order = list(range(count)) if ccw else list(range(count))[::-1]
    working = list(order)
    triangles: list[tuple[int, int, int]] = []
    guard = 0

    while len(working) > 3 and guard < 4 * count:
        guard += 1
        clipped = False
        for i in range(len(working)):
            prev_i = working[i - 1]
            curr_i = working[i]
            next_i = working[(i + 1) % len(working)]
            a, b, c = polygon[prev_i], polygon[curr_i], polygon[next_i]

            # Convex corner in a CCW ring?
            if _cross(a, b, c) <= 1e-12:
                continue
            # No other vertex inside the candidate ear?
            blocked = False
            for other in working:
                if other in (prev_i, curr_i, next_i):
                    continue
                if _point_in_triangle(polygon[other], a, b, c):
                    blocked = True
                    break
            if blocked:
                continue

            triangles.append((prev_i, curr_i, next_i))
            working.pop(i)
            clipped = True
            break

        if not clipped:
            # Degenerate or self-intersecting input: fall back to a fan so the
            # caller still gets a closed surface rather than nothing.
            break

    if len(working) == 3:
        triangles.append((working[0], working[1], working[2]))
    elif len(working) > 3:
        for i in range(1, len(working) - 1):
            triangles.append((working[0], working[i], working[i + 1]))

    return triangles
