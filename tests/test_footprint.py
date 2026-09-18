"""Footprint cleanup keeps corners and does not force rectangles."""

import numpy as np

from satrecon.stages.footprint import _polygon_area, _regularize, _simplify


def _contour(points):
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


def test_simplify_preserves_l_shape():
    l_shape = [(0, 0), (100, 0), (100, 40), (60, 40), (60, 100), (0, 100)]
    dense = []
    for i in range(len(l_shape)):
        a, b = np.array(l_shape[i]), np.array(l_shape[(i + 1) % len(l_shape)])
        for t in np.linspace(0, 1, 25, endpoint=False):
            dense.append(tuple(a + (b - a) * t))
    simplified = _simplify(_contour(dense), {"simplify_fraction": 0.012})
    assert 6 <= len(simplified) <= 10, "an L-shape must not be reduced to a rectangle"
    assert _polygon_area(simplified) > 0


def test_simplify_removes_redundant_vertices():
    dense = [(x, 0) for x in range(0, 101, 2)] + [(100, y) for y in range(0, 51, 2)] \
        + [(x, 50) for x in range(100, -1, -2)] + [(0, y) for y in range(50, -1, -2)]
    simplified = _simplify(_contour(dense), {"simplify_fraction": 0.012})
    assert len(simplified) <= 6


def test_regularize_moves_vertices_only_slightly():
    jagged = np.array([(0, 0), (100, 3), (98, 50), (-2, 47)], dtype=np.float64)
    cleaned = _regularize(jagged, angle_deg=0.0, tolerance_deg=12.0)
    assert cleaned.shape == jagged.shape
    assert np.max(np.abs(cleaned - jagged)) < 6.0


def test_regularize_leaves_angled_edges_alone():
    diamond = np.array([(50, 0), (100, 50), (50, 100), (0, 50)], dtype=np.float64)
    cleaned = _regularize(diamond, angle_deg=0.0, tolerance_deg=12.0)
    assert np.allclose(cleaned, diamond)


def test_polygon_area():
    assert _polygon_area(np.array([(0, 0), (10, 0), (10, 5), (0, 5)], dtype=np.float64)) == 25.0 * 2
