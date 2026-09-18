"""Stage 6 - scale resolution.

Rule: a scale is used only if it can be traced to something the user supplied.
There is no image-only fallback that quietly invents metres, because a wrong
scale silently corrupts every downstream dimension, height and floor count.

Resolution order (first match wins):
  1. scale.meters_per_pixel        - explicit, confidence 1.00
  2. scale.reference               - two points + a known distance, 0.90
  3. scale.ground_sample_distance  - product metadata, 0.85
  4. unknown                       - dimensions stay in pixels
"""

from __future__ import annotations

import math

from ..logging_setup import get_logger
from ..model import ScaleInfo

log = get_logger(__name__)

UNKNOWN_NOTE = "Scale unknown - dimensions are relative estimates."


def resolve(config_scale: dict) -> ScaleInfo:
    explicit = config_scale.get("meters_per_pixel")
    if explicit:
        value = float(explicit)
        if value <= 0:
            raise ValueError("scale.meters_per_pixel must be positive")
        log.info("scale from explicit user value: %.5f m/px", value)
        return ScaleInfo(value, "user_explicit", 1.0, "Scale supplied directly by the user.")

    reference = config_scale.get("reference") or {}
    point_a, point_b = reference.get("point_a"), reference.get("point_b")
    distance = reference.get("distance_meters")
    if point_a and point_b and distance:
        pixels = math.dist((float(point_a[0]), float(point_a[1])),
                           (float(point_b[0]), float(point_b[1])))
        if pixels <= 0:
            raise ValueError("scale.reference points must be distinct")
        if float(distance) <= 0:
            raise ValueError("scale.reference.distance_meters must be positive")
        value = float(distance) / pixels
        log.info(
            "scale from user reference: %.1f px = %.1f m -> %.5f m/px",
            pixels, float(distance), value,
        )
        return ScaleInfo(
            value, "user_reference", 0.90,
            f"Derived from a user reference of {distance} m over {pixels:.1f} px.",
        )

    gsd = config_scale.get("ground_sample_distance")
    if gsd:
        value = float(gsd)
        if value <= 0:
            raise ValueError("scale.ground_sample_distance must be positive")
        log.info("scale from ground sample distance: %.5f m/px", value)
        return ScaleInfo(
            value, "gsd", 0.85, "Ground sample distance supplied with the imagery."
        )

    log.warning("SCALE UNKNOWN - all dimensions will be reported in pixels")
    return ScaleInfo(None, "unknown", 0.0, UNKNOWN_NOTE)
