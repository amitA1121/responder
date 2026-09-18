"""Data model round-trips and confidence reporting."""

from satrecon.model import Building, Confidence, Scene, SceneMeta, ScaleInfo, result_hash


def _scene(tmp_path):
    building = Building(
        id="B001",
        footprint=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        bounding_box=(0, 0, 10, 5),
        area_px=50.0,
        confidence=Confidence(footprint=0.8, scale=0.0),
    )
    meta = SceneMeta(
        name="t", image_path="input/site.jpg", image_sha256="abc",
        image_width=100, image_height=100, config_fingerprint="f", detector="classical",
    )
    return Scene(meta=meta, scale=ScaleInfo(), buildings=[building])


def test_phase1_leaves_height_fields_null(tmp_path):
    scene = _scene(tmp_path)
    raw = scene.to_json()["buildings"][0]
    assert raw["estimated_height_m"] is None
    assert raw["estimated_floors"] is None
    assert raw["height_source"] == "not_estimated"
    assert raw["roof_type"] == "unknown"


def test_round_trip(tmp_path):
    path = _scene(tmp_path).save(tmp_path / "s.json")
    loaded = Scene.load(path)
    assert len(loaded.buildings) == 1
    assert loaded.buildings[0].id == "B001"
    assert loaded.buildings[0].footprint[2] == (10.0, 5.0)
    assert loaded.scale.known is False


def test_confidence_bands():
    assert Confidence(0.9, 0.9).band() == "HIGH"
    assert Confidence(0.5, 0.5).band() == "MEDIUM"
    assert Confidence(0.2, 0.1).band() == "LOW"
    # Unset aspects must not drag the average toward zero.
    assert Confidence(footprint=0.8, scale=0.8, height=None).overall() == 0.8


def test_unknown_scale_is_excluded_not_zeroed():
    """An unknown scale must not make a good footprint look unreliable."""
    not_attempted = Confidence(footprint=0.8, scale=None)
    attempted_badly = Confidence(footprint=0.8, scale=0.0)
    assert not_attempted.overall() == 0.8
    assert not_attempted.band() == "HIGH"
    assert attempted_badly.overall() == 0.4
    assert attempted_badly.band() == "MEDIUM"


def test_result_hash_ignores_timestamp():
    scene_a, scene_b = _scene(None), _scene(None)
    scene_b.meta.generated_at = "different"
    assert result_hash(scene_a.buildings, scene_a.scale) == result_hash(
        scene_b.buildings, scene_b.scale
    )
