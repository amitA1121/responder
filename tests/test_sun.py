"""Solar geometry: the only source of the shadow-to-height multiplier.

Like scale, an unknown sun position must stay unknown rather than default to
something plausible.
"""

import datetime as dt
import math

import pytest

from satrecon.stages import sun


UTC = dt.timezone.utc


# -- resolution rules ------------------------------------------------------

def test_unknown_when_nothing_supplied():
    position = sun.resolve({}, {})
    assert position.elevation_deg is None
    assert position.known is False
    assert position.source == "unknown"
    assert position.shadow_ratio() is None


def test_explicit_elevation_wins_over_geo():
    position = sun.resolve(
        {"elevation_deg": 60.0},
        {"center_lat": 51.5, "center_lon": 0.0},
    )
    assert position.elevation_deg == 60.0
    assert position.source == "user_explicit"
    assert position.confidence == 1.0
    assert position.shadow_ratio() == pytest.approx(math.tan(math.radians(60.0)))


def test_computed_from_geo_and_timestamp():
    position = sun.resolve(
        {"timestamp": "2024-06-21T09:00:00+00:00"},
        {"center_lat": 51.5, "center_lon": 0.0},
    )
    assert position.source == "computed"
    assert position.known is True
    assert position.azimuth_deg is not None


def test_geo_without_timestamp_stays_unknown():
    position = sun.resolve({}, {"center_lat": 51.5, "center_lon": 0.0})
    assert position.known is False
    assert position.source == "unknown"


def test_night_time_is_reported_unknown_not_negative():
    """A below-horizon sun means the inputs are wrong, not that heights are negative."""
    position = sun.resolve(
        {"timestamp": "2024-06-21T00:00:00+00:00"},
        {"center_lat": 51.5, "center_lon": 0.0},
    )
    assert position.known is False
    assert position.elevation_deg is None
    assert "below the horizon" in position.note


@pytest.mark.parametrize("bad", [{"elevation_deg": 0.0}, {"elevation_deg": 90.0}, {"elevation_deg": -5}])
def test_invalid_elevation_raises(bad):
    with pytest.raises(ValueError):
        sun.resolve(bad, {})


def test_naive_timestamp_is_rejected():
    """A local time with no offset cannot be placed on the sun's path."""
    with pytest.raises(ValueError, match="UTC offset"):
        sun.resolve({"timestamp": "2024-06-21T09:00:00"}, {"center_lat": 32.0, "center_lon": 34.8})


def test_unparseable_timestamp_raises():
    with pytest.raises(ValueError, match="ISO 8601"):
        sun.resolve({"timestamp": "yesterday"}, {"center_lat": 32.0, "center_lon": 34.8})


# -- the astronomy itself --------------------------------------------------

def test_overhead_at_equator_on_equinox_noon():
    # 12:00 UTC on the equinox date is not the equinox instant (03:06 UTC in
    # 2024) nor exactly solar noon (equation of time is ~ -7 min), so the sun
    # is close to but not at the zenith. ~88 deg is the physically correct
    # answer; anything much lower would signal a real error.
    elevation, _ = sun.solar_position(0.0, 0.0, dt.datetime(2024, 3, 20, 12, 0, tzinfo=UTC))
    assert elevation == pytest.approx(88.0, abs=1.0)


def test_elevation_is_symmetric_about_solar_noon():
    before, _ = sun.solar_position(45.0, 0.0, dt.datetime(2024, 6, 21, 9, 0, tzinfo=UTC))
    after, _ = sun.solar_position(45.0, 0.0, dt.datetime(2024, 6, 21, 15, 0, tzinfo=UTC))
    assert before == pytest.approx(after, abs=0.6)


def test_solstice_noon_elevation_matches_declination():
    """At solar noon, elevation = 90 - |latitude - declination|; 23.44 deg in June."""
    elevation, azimuth = sun.solar_position(40.0, 0.0, dt.datetime(2024, 6, 21, 12, 0, tzinfo=UTC))
    assert elevation == pytest.approx(90.0 - (40.0 - 23.44), abs=1.0)
    # Northern mid-latitude noon: the sun is due south.
    assert azimuth == pytest.approx(180.0, abs=2.0)


def test_azimuth_moves_east_to_west_through_the_day():
    morning = sun.solar_position(32.0, 34.8, dt.datetime(2024, 6, 21, 4, 0, tzinfo=UTC))[1]
    noonish = sun.solar_position(32.0, 34.8, dt.datetime(2024, 6, 21, 9, 40, tzinfo=UTC))[1]
    evening = sun.solar_position(32.0, 34.8, dt.datetime(2024, 6, 21, 15, 0, tzinfo=UTC))[1]
    assert morning < noonish < evening
    assert morning < 120.0      # east-ish
    assert evening > 240.0      # west-ish


def test_timezone_aware_conversion():
    """The same instant expressed in two zones must give the same answer."""
    utc = dt.datetime(2024, 6, 21, 9, 0, tzinfo=UTC)
    plus_three = dt.datetime(2024, 6, 21, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
    assert sun.solar_position(32.0, 34.8, utc) == pytest.approx(
        sun.solar_position(32.0, 34.8, plus_three)
    )


def test_naive_datetime_rejected_at_the_astronomy_layer():
    with pytest.raises(ValueError):
        sun.solar_position(32.0, 34.8, dt.datetime(2024, 6, 21, 9, 0))


# -- image-frame cross-check ----------------------------------------------

def test_expected_shadow_direction_needs_north_and_azimuth():
    known = sun.SunPosition(45.0, 180.0, "computed", 0.9)
    assert sun.expected_shadow_direction_deg(known, None) is None
    assert sun.expected_shadow_direction_deg(sun.SunPosition(45.0, None), 0.0) is None


def test_shadow_points_away_from_the_sun():
    """Sun due south (az 180) in a north-up image: shadows fall toward image-up."""
    position = sun.SunPosition(45.0, 180.0, "computed", 0.9)
    angle = sun.expected_shadow_direction_deg(position, 0.0)
    # Image frame is CCW from +x with y pointing down, so "up" is 270 deg.
    assert angle == pytest.approx(270.0, abs=0.5)
