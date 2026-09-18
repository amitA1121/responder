"""End-to-end behaviour on a small synthetic scene.

The synthetic image keeps the test fast and independent of any real imagery:
three bright rectangles with shadows on a textured ground, crossed by a road.
"""

import cv2
import numpy as np
import pytest

from satrecon import pipeline
from satrecon.config import Config
from satrecon.model import Scene


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(7)
    size = 480
    # Signed arithmetic before the uint8 cast: a uint8 add would wrap negative
    # noise round to ~250 and flood the image with white.
    ground = np.full((size, size, 3), 118, np.int16)
    ground += rng.integers(-9, 10, (size, size, 3), dtype=np.int16)
    image = np.clip(ground, 0, 255).astype(np.uint8)

    # A road: bright but thin and elongated.
    cv2.rectangle(image, (0, 228), (size, 252), (132, 132, 132), -1)

    boxes = [(40, 40, 150, 130), (250, 60, 380, 190), (70, 300, 220, 430)]
    for x1, y1, x2, y2 in boxes:
        # Shadow first, offset down-left, then the bright roof over it.
        cv2.rectangle(image, (x1 - 18, y1 + 18), (x2 - 18, y2 + 18), (48, 48, 48), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (208, 206, 202), -1)

    path = tmp_path_factory.mktemp("img") / "synthetic.png"
    cv2.imwrite(str(path), image)
    return path, boxes


def test_detects_synthetic_buildings(synthetic):
    path, boxes = synthetic
    result = pipeline.analyze(path, Config.load(None))
    assert len(result.scene.buildings) >= len(boxes)

    # Each planted rectangle should have a detection centred near it.
    centres = [
        (b.bounding_box[0] + b.bounding_box[2] / 2, b.bounding_box[1] + b.bounding_box[3] / 2)
        for b in result.scene.buildings
    ]
    for x1, y1, x2, y2 in boxes:
        target = ((x1 + x2) / 2, (y1 + y2) / 2)
        nearest = min(np.hypot(cx - target[0], cy - target[1]) for cx, cy in centres)
        assert nearest < 60, f"no detection near planted building {target}"


def test_scale_unknown_keeps_metres_null(synthetic):
    path, _ = synthetic
    scene = pipeline.analyze(path, Config.load(None)).scene
    assert scene.scale.known is False
    assert all(b.length_m is None and b.area_m2 is None for b in scene.buildings)
    assert any("Scale unknown" in note for note in scene.meta.notes)


def test_scale_applied_when_supplied(synthetic, tmp_path):
    path, _ = synthetic
    config = Config.load(None)
    config._data["scale"]["meters_per_pixel"] = 0.5
    scene = pipeline.analyze(path, config).scene
    assert scene.scale.known
    for building in scene.buildings:
        # Stored values are rounded to centimetres.
        assert building.length_m == pytest.approx(building.length_px * 0.5, abs=0.01)


def test_reproducible(synthetic):
    path, _ = synthetic
    first = pipeline.analyze(path, Config.load(None)).scene
    second = pipeline.analyze(path, Config.load(None)).scene
    assert first.meta.result_hash == second.meta.result_hash
    assert first.meta.image_sha256 == second.meta.image_sha256


def test_scene_file_round_trip(synthetic, tmp_path):
    path, _ = synthetic
    scene = pipeline.analyze(path, Config.load(None)).scene
    reloaded = Scene.load(scene.save(tmp_path / "scene.json"))
    assert len(reloaded.buildings) == len(scene.buildings)
    assert reloaded.meta.result_hash == scene.meta.result_hash
