"""End-to-end Phase 2: a synthetic image runs image -> ... -> heights -> GLB.

The image is generated in code with a bright rectangular 'building' and an
attached dark shadow of a known length, at a known scale and sun elevation, so
the recovered height can be checked against ground truth the test itself set.
This is the honest way to validate the geometry: real imagery has no known
answer to compare against.
"""

import struct

import cv2
import numpy as np
import pytest

from satrecon.config import Config
from satrecon.geometry import export, mesh
from satrecon.pipeline import analyze


# The building is 170x180 px; its shadow extends 60 px to the right. At
# 0.5 m/px that is a 30 m ground shadow, and a 45 deg sun (tan 45 = 1) implies
# a 30 m building. That number is the ground truth these tests check against.
GROUND_TRUTH_HEIGHT_M = 30.0


def _make_scene_image(path, size=500):
    # Neutral, lightly textured ground: not vegetation, not shadow, not roof.
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), 120, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (150, 150), (320, 330), (210, 210, 205), -1)   # bright roof
    cv2.rectangle(img, (320, 150), (380, 330), (28, 28, 28), -1)      # 60 px shadow, sun from the left
    cv2.imwrite(str(path), img)
    return path


_FOOTPRINT = {
    "min_area_px": 500, "simplify_fraction": 0.02, "min_vertices": 4, "max_vertices": 40,
    "regularize": True, "regularize_angle_tolerance_deg": 12.0, "fill_hole_area_px": 250,
}


def _config(scale=True, sun=True):
    data = {**Config.load(None).as_dict(), "footprint": _FOOTPRINT}
    if scale:
        data["scale"] = {"meters_per_pixel": 0.5}
    if sun:
        data["sun"] = {"elevation_deg": 45.0, "azimuth_deg": None, "timestamp": None}
    return Config(data)


def _measured(scene):
    return [b for b in scene.buildings if b.estimated_height_m is not None]


def test_full_pipeline_recovers_the_true_height(tmp_path):
    image_path = _make_scene_image(tmp_path / "synthetic.png")
    scene = analyze(image_path, _config(), "synthetic").scene

    # The pipeline ran to completion and recorded the sun it used.
    assert scene.sun.elevation_deg == 45.0
    assert scene.sun.source == "user_explicit"
    assert scene.scale.known

    assert len(scene.buildings) >= 1
    measured = _measured(scene)
    assert measured, "expected at least one shadow-derived height"

    # The tallest measured building should match the 30 m we drew, to a couple
    # of metres - the tolerance the shadow measurement can honestly claim.
    tallest = max(b.estimated_height_m for b in measured)
    assert tallest == pytest.approx(GROUND_TRUTH_HEIGHT_M, abs=3.0)

    for b in measured:
        assert b.height_source == "shadow"
        assert b.estimated_floors is not None and b.estimated_floors >= 1
        assert b.confidence.height is not None


def test_full_pipeline_exports_valid_glb(tmp_path):
    image_path = _make_scene_image(tmp_path / "synthetic.png")
    scene = analyze(image_path, _config(), "synthetic").scene
    model_mesh = mesh.scene_to_mesh(scene.buildings, scene.scale)
    assert not model_mesh.is_empty

    data = export.write_glb(model_mesh, tmp_path / "model.glb", units="meters").read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    # The building's true height survives into the geometry.
    assert model_mesh.vertices[:, 1].max() == pytest.approx(
        max(b.estimated_height_m for b in _measured(scene)), abs=0.01
    )


def test_pipeline_without_sun_leaves_heights_null(tmp_path):
    image_path = _make_scene_image(tmp_path / "synthetic.png")
    scene = analyze(image_path, _config(sun=False), "synthetic").scene
    assert scene.sun.elevation_deg is None
    assert scene.buildings, "detector should still find the building"
    assert all(b.estimated_height_m is None for b in scene.buildings)
    assert all(b.height_source == "not_estimated" for b in scene.buildings)

    # Geometry still exports, using placeholder heights.
    model_mesh = mesh.scene_to_mesh(scene.buildings, scene.scale)
    assert model_mesh.vertices[:, 1].max() == pytest.approx(3.0)


def test_reproducibility_hash_includes_height(tmp_path):
    image_path = _make_scene_image(tmp_path / "synthetic.png")
    cfg = _config()
    a = analyze(image_path, cfg, "synthetic").scene
    b = analyze(image_path, cfg, "synthetic").scene
    assert a.meta.result_hash == b.meta.result_hash
    # And the height genuinely participates: zeroing it changes the hash.
    from satrecon.model import result_hash
    hashed_with = a.meta.result_hash
    for building in a.buildings:
        building.estimated_height_m = None
    assert result_hash(a.buildings, a.scale) != hashed_with
