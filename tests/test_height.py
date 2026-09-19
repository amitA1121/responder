"""Height from shadows.

The pure geometry is tested against hand-computed values; the image
measurement is tested against synthetic scenes where the true shadow length is
known exactly, so a wrong answer means a real regression, not image noise.
"""

import math

import numpy as np
import pytest

from satrecon.model import Building, Confidence, ScaleInfo
from satrecon.stages import height
from satrecon.stages.shadow import ShadowDirection
from satrecon.stages.sun import SunPosition


def _sun(elevation):
    return SunPosition(elevation, None, "user_explicit", 1.0)


def _rect_building(x, y, w, h, ident="B001"):
    footprint = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Building(id=ident, footprint=footprint, bounding_box=(x, y, w, h),
                    area_px=float(w * h), confidence=Confidence(footprint=0.9, scale=1.0))


# -- pure geometry ---------------------------------------------------------

def test_height_at_45_degrees_equals_shadow_length():
    # tan(45) = 1, so height equals shadow length exactly.
    assert height.height_from_shadow_m(20.0, _sun(45.0)) == pytest.approx(20.0)


def test_height_scales_with_tangent_of_elevation():
    assert height.height_from_shadow_m(10.0, _sun(30.0)) == pytest.approx(10.0 * math.tan(math.radians(30)))
    assert height.height_from_shadow_m(10.0, _sun(60.0)) == pytest.approx(10.0 * math.tan(math.radians(60)))


def test_unknown_sun_yields_no_height():
    assert height.height_from_shadow_m(20.0, SunPosition()) is None


def test_floors_round_to_nearest_and_floor_at_one():
    assert height.estimated_floors(9.6, 3.2) == 3
    assert height.estimated_floors(10.5, 3.2) == 3
    assert height.estimated_floors(1.0, 3.2) == 1     # never zero for a real building
    assert height.estimated_floors(None, 3.2) is None


# -- image measurement on synthetic scenes ---------------------------------

def _shadow_scene(size=200):
    return np.zeros((size, size), dtype=np.float32)


def test_measures_a_known_horizontal_shadow():
    """A 40x40 footprint casts a 30 px shadow straight to the right (0 deg)."""
    shadow = _shadow_scene()
    building_x, building_y, side = 60, 80, 40
    shadow_len = 30
    # Shadow band on the right side of the building.
    shadow[building_y:building_y + side, building_x + side:building_x + side + shadow_len] = 1.0

    polygon = np.array(_rect_building(building_x, building_y, side, side).footprint)
    length, rays, agreement = height.measure_shadow_length_px(
        polygon, shadow, direction_angle_deg=0.0, params={}
    )
    assert rays >= 3
    assert length == pytest.approx(shadow_len, abs=2.0)
    assert agreement > 0.8      # a clean rectangular shadow: rays agree


def test_measures_a_known_downward_shadow():
    """Same building, shadow pointing down the image (90 deg)."""
    shadow = _shadow_scene()
    bx, by, side, shadow_len = 60, 60, 40, 25
    shadow[by + side:by + side + shadow_len, bx:bx + side] = 1.0

    polygon = np.array(_rect_building(bx, by, side, side).footprint)
    length, rays, _ = height.measure_shadow_length_px(polygon, shadow, 90.0, {})
    assert length == pytest.approx(shadow_len, abs=2.0)


def test_no_shadow_returns_none():
    shadow = _shadow_scene()
    polygon = np.array(_rect_building(60, 60, 40, 40).footprint)
    length, rays, agreement = height.measure_shadow_length_px(polygon, shadow, 0.0, {})
    assert length is None
    assert agreement == 0.0


def test_end_to_end_height_on_synthetic_scene():
    """Full path: 30 px shadow, 0.5 m/px, sun at 45 deg -> 15 m, ~5 floors."""
    shadow = _shadow_scene()
    bx, by, side, shadow_len = 60, 80, 40, 30
    shadow[by:by + side, bx + side:bx + side + shadow_len] = 1.0

    building = _rect_building(bx, by, side, side)
    scale = ScaleInfo(0.5, "user_explicit", 1.0)
    direction = ShadowDirection(0.0, 20.0, 0.5, 0.9, "estimated")
    sun = _sun(45.0)

    estimates = height.estimate(
        [building], shadow, direction, scale, sun, scale_factor=1.0, params={},
    )
    est = estimates[0]
    # 30 px * 0.5 m/px = 15 m of shadow; tan(45)=1 -> 15 m tall.
    assert est.height_m == pytest.approx(15.0, abs=1.0)
    assert building.estimated_height_m == est.height_m
    # Floors follow the measured height, not the idealised 15 m.
    assert building.estimated_floors == max(round(est.height_m / 3.2), 1)
    assert building.height_source == "shadow"
    assert 0.0 < building.confidence.height <= 0.95


def test_unknown_scale_leaves_height_none():
    shadow = _shadow_scene()
    building = _rect_building(60, 80, 40, 40)
    direction = ShadowDirection(0.0, 20.0, 0.5, 0.9, "estimated")
    estimates = height.estimate(
        [building], shadow, direction, ScaleInfo(), _sun(45.0), 1.0, {},
    )
    assert estimates[0].height_m is None
    assert building.estimated_height_m is None
    assert building.height_source == "not_estimated"
    assert building.confidence.height is None


def test_unknown_sun_leaves_height_none():
    shadow = _shadow_scene()
    bx, by, side = 60, 80, 40
    shadow[by:by + side, bx + side:bx + side + 30] = 1.0
    building = _rect_building(bx, by, side, side)
    direction = ShadowDirection(0.0, 20.0, 0.5, 0.9, "estimated")
    estimates = height.estimate(
        [building], shadow, direction, ScaleInfo(0.5, "user_explicit", 1.0),
        SunPosition(), 1.0, {},
    )
    assert estimates[0].height_m is None
    assert building.height_source == "not_estimated"


def test_user_edited_height_is_not_overwritten():
    shadow = _shadow_scene()
    bx, by, side = 60, 80, 40
    shadow[by:by + side, bx + side:bx + side + 30] = 1.0
    building = _rect_building(bx, by, side, side)
    building.user_edited = True
    building.estimated_height_m = 99.0
    building.confidence.height = 1.0
    direction = ShadowDirection(0.0, 20.0, 0.5, 0.9, "estimated")
    height.estimate(
        [building], shadow, direction, ScaleInfo(0.5, "user_explicit", 1.0),
        _sun(45.0), 1.0, {},
    )
    assert building.estimated_height_m == 99.0      # human value preserved


def test_scale_factor_maps_footprint_into_working_space():
    """Footprints are source pixels; the shadow mask is working resolution."""
    factor = 0.5
    shadow = _shadow_scene(size=200)
    # Building in *working* pixels, then its source footprint is 2x larger.
    bx_w, by_w, side_w, shadow_len_w = 40, 40, 20, 20
    shadow[by_w:by_w + side_w, bx_w + side_w:bx_w + side_w + shadow_len_w] = 1.0
    source_building = _rect_building(int(bx_w / factor), int(by_w / factor),
                                     int(side_w / factor), int(side_w / factor))
    scale = ScaleInfo(0.5, "user_explicit", 1.0)   # metres per SOURCE pixel
    direction = ShadowDirection(0.0, 20.0, 0.5, 0.9, "estimated")
    estimates = height.estimate(
        [source_building], shadow, direction, scale, _sun(45.0),
        scale_factor=factor, params={},
    )
    # 20 working px = 40 source px = 20 m shadow -> 20 m tall at 45 deg.
    assert estimates[0].height_m == pytest.approx(20.0, abs=2.0)
