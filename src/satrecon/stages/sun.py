"""Stage 7 - solar geometry.

Shadow length only becomes a height once the sun's elevation is known:

    height = shadow_length * tan(elevation)

The elevation is computed from where and when the image was taken (NOAA solar
position algorithm), never guessed from the image.  Same rule as `scale.py`:
if the inputs are absent the answer is UNKNOWN and nothing downstream invents
a number.

Resolution order (first match wins):
  1. sun.elevation_deg   - stated by the user, confidence 1.00
  2. geo + timestamp     - computed from lat/lon/UTC time, 0.90
  3. unknown             - heights are not estimated
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SunPosition:
    """Solar elevation and azimuth, in degrees.

    `elevation_deg is None` means unknown.  `azimuth_deg` is measured
    clockwise from true north and is only available when the position was
    computed from coordinates; it is used to sanity-check the shadow
    direction recovered from the image, not to override it.
    """

    elevation_deg: float | None = None
    azimuth_deg: float | None = None
    source: str = "unknown"      # user_explicit | computed | unknown
    confidence: float = 0.0
    note: str = "Sun elevation unknown - heights are not estimated."

    @property
    def known(self) -> bool:
        return self.elevation_deg is not None and 0.0 < self.elevation_deg < 90.0

    def shadow_ratio(self) -> float | None:
        """Multiplier from shadow length to height: tan(elevation)."""
        if not self.known:
            return None
        return math.tan(math.radians(self.elevation_deg))


def solar_position(lat: float, lon: float, when: _dt.datetime) -> tuple[float, float]:
    """Sun elevation and azimuth (degrees) for a place and an instant.

    NOAA solar position algorithm.  `when` must be timezone-aware; it is
    converted to UTC internally.  Accurate to roughly 0.1 deg, which is far
    below the error of measuring a shadow in a satellite image.
    """
    if when.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (e.g. ...+02:00 or Z)")
    utc = when.astimezone(_dt.timezone.utc)

    day_of_year = int(utc.strftime("%j"))
    hours = utc.hour + utc.minute / 60.0 + utc.second / 3600.0

    # Fractional year, radians.
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hours - 12.0) / 24.0)

    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.001480 * math.sin(3 * gamma)
    )

    # True solar time, minutes. Working in UTC makes the zone term vanish.
    true_solar_minutes = (hours * 60.0) + eq_time + 4.0 * lon
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)

    lat_rad = math.radians(lat)
    cos_zenith = (
        math.sin(lat_rad) * math.sin(declination)
        + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = 90.0 - math.degrees(zenith)

    sin_zenith = math.sin(zenith)
    if abs(sin_zenith) < 1e-9:
        azimuth = 0.0
    else:
        cos_azimuth = (
            math.sin(lat_rad) * cos_zenith - math.sin(declination)
        ) / (math.cos(lat_rad) * sin_zenith)
        cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))
        # acos loses the sign of the hour angle. Afternoon (west) bearings are
        # az + 180; morning (east) bearings mirror onto 540 - az (NOAA).
        azimuth = (azimuth + 180.0) % 360.0 if hour_angle > 0 else (540.0 - azimuth) % 360.0
    return elevation, azimuth


def resolve(sun_config: dict, geo_config: dict) -> SunPosition:
    """Decide the sun elevation to use, preferring what the user stated."""
    explicit = sun_config.get("elevation_deg")
    if explicit is not None:
        value = float(explicit)
        if not 0.0 < value < 90.0:
            raise ValueError("sun.elevation_deg must be between 0 and 90 (exclusive)")
        log.info("sun elevation from explicit user value: %.2f deg", value)
        azimuth = sun_config.get("azimuth_deg")
        return SunPosition(
            value,
            float(azimuth) if azimuth is not None else None,
            "user_explicit",
            1.0,
            "Sun elevation supplied directly by the user.",
        )

    lat, lon = geo_config.get("center_lat"), geo_config.get("center_lon")
    stamp = sun_config.get("timestamp")
    if lat is not None and lon is not None and stamp:
        when = _parse_timestamp(stamp)
        elevation, azimuth = solar_position(float(lat), float(lon), when)
        if elevation <= 0.0:
            log.warning(
                "computed sun elevation is %.2f deg (sun below horizon) - "
                "check geo.center_lat/lon and sun.timestamp", elevation,
            )
            return SunPosition(
                None, None, "unknown", 0.0,
                f"Computed sun elevation {elevation:.1f} deg is below the horizon; "
                "the timestamp or coordinates are wrong. Heights not estimated.",
            )
        log.info(
            "sun elevation %.2f deg, azimuth %.2f deg computed for %.4f,%.4f at %s",
            elevation, azimuth, float(lat), float(lon), when.isoformat(),
        )
        return SunPosition(
            elevation, azimuth, "computed", 0.90,
            f"Computed from {lat},{lon} at {when.isoformat()}.",
        )

    log.warning("SUN ELEVATION UNKNOWN - heights will not be estimated")
    return SunPosition()


def _parse_timestamp(value: str | _dt.datetime) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        when = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            when = _dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"sun.timestamp is not an ISO 8601 datetime: {value!r}"
            ) from exc
    if when.tzinfo is None:
        raise ValueError(
            "sun.timestamp must include a UTC offset (e.g. 2024-06-01T10:30:00+03:00); "
            "a local time without one cannot be placed on the sun's path"
        )
    return when


def expected_shadow_direction_deg(
    sun: SunPosition, north_offset_deg: float | None
) -> float | None:
    """Where shadows should point in the image, if the sun position is known.

    Shadows fall opposite the sun.  Image frame, degrees CCW from +x, which is
    the convention `shadow.py` uses.  Returns None when the azimuth or the
    image's north direction is unknown - in that case the image-derived shadow
    direction stands on its own with nothing to check it against.
    """
    if sun.azimuth_deg is None or north_offset_deg is None:
        return None
    # Shadows fall opposite the sun, so the shadow bearing is the azimuth + 180.
    shadow_bearing = (float(sun.azimuth_deg) + 180.0) % 360.0
    # Map a compass bearing (clockwise from true north) to shadow.py's image
    # angle (CCW from +x with y pointing down, so image-up is 270 deg). With a
    # north-up image, north(bearing 0)->up(270), east(90)->right(0): phi = b-90.
    # A positive north_offset turns true north anticlockwise from image-up,
    # which is the -phi direction, so it subtracts.
    return (shadow_bearing - 90.0 - float(north_offset_deg)) % 360.0
