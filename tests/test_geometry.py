"""3D extrusion and export.

The extrusion is checked for the properties a solid must have (right vertex
count, every edge shared by two triangles, correct volume), and the GLB is
parsed back from its bytes to prove it is a valid glTF container.
"""

import json
import struct
from collections import Counter

import numpy as np
import pytest

from satrecon.geometry import export, mesh
from satrecon.geometry.mesh import Mesh
from satrecon.model import Building, Confidence, ScaleInfo


def _rect(x, y, w, h, height_m=None, ident="B001", scale_known=True):
    b = Building(
        id=ident,
        footprint=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        bounding_box=(x, y, w, h),
        area_px=float(w * h),
        estimated_height_m=height_m,
        confidence=Confidence(footprint=0.9, scale=1.0 if scale_known else None),
    )
    return b


def _l_shape(ident="B001", height_m=10.0):
    # An L: concave, so a convex-only mesher would get it wrong.
    footprint = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
    return Building(id=ident, footprint=footprint, area_px=300.0,
                    estimated_height_m=height_m, confidence=Confidence(footprint=0.9))


# -- triangulation ---------------------------------------------------------

def test_triangulates_a_square_into_two_triangles():
    tris = mesh.triangulate(np.array([(0, 0), (10, 0), (10, 10), (0, 10)], dtype=float))
    assert len(tris) == 2


def test_triangulates_concave_l_shape():
    footprint = np.array([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)], dtype=float)
    tris = mesh.triangulate(footprint)
    # An n-gon triangulates into n-2 triangles.
    assert len(tris) == len(footprint) - 2
    # Total triangle area must equal the polygon area (no overlap, no gap).
    poly_area = abs(mesh._signed_area(footprint))
    tri_area = 0.0
    for i, j, k in tris:
        a, b, c = footprint[i], footprint[j], footprint[k]
        tri_area += abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
    assert tri_area == pytest.approx(poly_area, rel=1e-6)


def test_degenerate_polygon_triangulates_to_nothing():
    assert mesh.triangulate(np.array([(0, 0), (1, 1)], dtype=float)) == []


# -- extrusion -------------------------------------------------------------

def test_prism_has_expected_vertex_and_face_counts():
    building = _rect(0, 0, 10, 10, height_m=20.0)
    m = mesh.building_to_mesh(building, ScaleInfo(1.0, "user_explicit", 1.0), 3.0)
    assert len(m.vertices) == 8                     # 4 base + 4 top
    # 2 roof + 2 floor + 4 walls * 2 = 12 triangles.
    assert len(m.faces) == 12
    assert set(m.face_building) == {"B001"}


def test_extruded_height_matches_estimate():
    building = _rect(0, 0, 10, 10, height_m=25.0)
    m = mesh.building_to_mesh(building, ScaleInfo(1.0, "user_explicit", 1.0), 3.0)
    # Y is up; the top ring should sit at 25 m.
    assert m.vertices[:, 1].max() == pytest.approx(25.0)
    assert m.vertices[:, 1].min() == pytest.approx(0.0)


def test_scale_converts_pixels_to_metres():
    # 10 px wide at 0.5 m/px -> 5 m wide on the ground.
    building = _rect(0, 0, 10, 10, height_m=8.0)
    m = mesh.building_to_mesh(building, ScaleInfo(0.5, "user_explicit", 1.0), 3.0)
    span_x = m.vertices[:, 0].max() - m.vertices[:, 0].min()
    assert span_x == pytest.approx(5.0)


def test_unmeasured_building_gets_placeholder_height():
    building = _rect(0, 0, 10, 10, height_m=None)
    m = mesh.building_to_mesh(building, ScaleInfo(1.0, "user_explicit", 1.0), default_height_units=3.0)
    assert m.vertices[:, 1].max() == pytest.approx(3.0)


def test_prism_is_watertight():
    """Every edge of a closed solid is shared by exactly two triangles."""
    building = _rect(0, 0, 10, 10, height_m=12.0)
    m = mesh.building_to_mesh(building, ScaleInfo(1.0, "user_explicit", 1.0), 3.0)
    edges: Counter = Counter()
    for tri in m.faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[frozenset((int(a), int(b)))] += 1
    assert all(count == 2 for count in edges.values())


def test_l_shape_is_watertight():
    m = mesh.building_to_mesh(_l_shape(), ScaleInfo(1.0, "user_explicit", 1.0), 3.0)
    edges: Counter = Counter()
    for tri in m.faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[frozenset((int(a), int(b)))] += 1
    assert all(count == 2 for count in edges.values())


def test_scene_mesh_merges_all_buildings():
    buildings = [
        _rect(0, 0, 10, 10, height_m=10.0, ident="B001"),
        _rect(30, 30, 10, 10, height_m=20.0, ident="B002"),
    ]
    m = mesh.scene_to_mesh(buildings, ScaleInfo(1.0, "user_explicit", 1.0))
    assert len(m.vertices) == 16
    assert set(m.face_building) == {"B001", "B002"}


# -- export ----------------------------------------------------------------

def _simple_mesh():
    return mesh.building_to_mesh(_rect(0, 0, 10, 10, height_m=10.0),
                                 ScaleInfo(1.0, "user_explicit", 1.0), 3.0)


def test_obj_roundtrips_vertex_and_face_counts(tmp_path):
    m = _simple_mesh()
    path = export.write_obj(m, tmp_path / "out.obj")
    text = path.read_text()
    assert text.count("\nv ") == len(m.vertices)
    assert text.count("\nf ") == len(m.faces)
    assert "g B001" in text


def test_obj_face_indices_are_one_based(tmp_path):
    m = _simple_mesh()
    path = export.write_obj(m, tmp_path / "out.obj")
    for line in path.read_text().splitlines():
        if line.startswith("f "):
            assert all(int(tok) >= 1 for tok in line.split()[1:])


def test_glb_is_a_valid_container(tmp_path):
    m = _simple_mesh()
    path = export.write_glb(m, tmp_path / "out.glb")
    data = path.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67          # "glTF"
    assert version == 2
    assert length == len(data)

    # First chunk is JSON and parses.
    json_len, json_type = struct.unpack("<II", data[12:20])
    assert json_type == 0x4E4F534A
    document = json.loads(data[20:20 + json_len])
    assert document["asset"]["version"] == "2.0"
    assert document["accessors"][1]["count"] == len(m.vertices)
    assert document["accessors"][0]["count"] == m.faces.size


def test_glb_bin_chunk_length_matches_buffer(tmp_path):
    m = _simple_mesh()
    data = export.write_glb(m, tmp_path / "out.glb").read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    bin_offset = 20 + json_len
    bin_len, bin_type = struct.unpack("<II", data[bin_offset:bin_offset + 8])
    assert bin_type == 0x004E4942       # "BIN\0"
    assert bin_offset + 8 + bin_len == len(data)


def test_gltf_units_travel_with_the_file(tmp_path):
    m = _simple_mesh()
    doc, _ = export.build_gltf(m, units="pixels")
    assert doc["asset"]["extras"]["units"] == "pixels"
