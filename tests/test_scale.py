"""Scale resolution must never invent metres."""

import pytest

from satrecon.stages import scale


def test_unknown_when_nothing_supplied():
    info = scale.resolve({})
    assert info.meters_per_pixel is None
    assert info.known is False
    assert info.source == "unknown"
    assert info.confidence == 0.0
    assert info.px_to_m(100) is None


def test_explicit_value_wins():
    info = scale.resolve({"meters_per_pixel": 0.25, "ground_sample_distance": 9.9})
    assert info.meters_per_pixel == 0.25
    assert info.source == "user_explicit"
    assert info.confidence == 1.0
    assert info.px_to_m(100) == pytest.approx(25.0)


def test_two_point_reference():
    info = scale.resolve(
        {"reference": {"point_a": [0, 0], "point_b": [0, 200], "distance_meters": 50}}
    )
    assert info.meters_per_pixel == pytest.approx(0.25)
    assert info.source == "user_reference"


def test_reference_beats_gsd():
    info = scale.resolve({
        "reference": {"point_a": [0, 0], "point_b": [100, 0], "distance_meters": 50},
        "ground_sample_distance": 1.0,
    })
    assert info.source == "user_reference"


def test_ground_sample_distance_fallback():
    info = scale.resolve({"ground_sample_distance": 0.31})
    assert info.meters_per_pixel == 0.31
    assert info.source == "gsd"


@pytest.mark.parametrize("bad", [
    {"meters_per_pixel": -1},
    {"ground_sample_distance": 0.0, "meters_per_pixel": -0.5},
    {"reference": {"point_a": [5, 5], "point_b": [5, 5], "distance_meters": 10}},
    {"reference": {"point_a": [0, 0], "point_b": [10, 0], "distance_meters": -3}},
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        scale.resolve(bad)
