"""Stage 8 - height estimation from cast shadows.

Given the sun's elevation, a shadow of length L on flat ground implies a
structure of height:

    H = L * tan(elevation)

The hard part is measuring L from the image. We march rays outward from the
shadow-facing edge of each footprint, along the recovered shadow direction,
and count how far the shadow cue persists. That measurement is noisy, so every
height carries a confidence built from three things: how much the rays agreed,
how trustworthy the shadow direction was, and how trustworthy the sun position
was.

Nothing here invents a height. If the sun elevation or the scale is unknown,
the height stays None and the building keeps its Phase 1 "not_estimated" state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..logging_setup import get_logger
from ..model import Building, ScaleInfo
from .sun import SunPosition

log = get_logger(__name__)


@dataclass
class HeightEstimate:
    """One building's height, with the evidence that produced it."""

    height_m: float | None
    shadow_length_px: float | None      # source pixels, robust aggregate
    ray_count: int
    ray_agreement: float                 # 1 - spread/median, in [0, 1]
    confidence: float
    source: str                          # "shadow" | "not_estimated" | "user_edited"
    note: str = ""


# -- pure geometry (no image) ---------------------------------------------

def height_from_shadow_m(shadow_length_m: float, sun: SunPosition) -> float | None:
    """Convert a ground shadow length in metres to a height in metres.

    Returns None if the sun position cannot supply a ratio, so callers can
    never accidentally treat a missing sun as a zero-height building.
    """
    ratio = sun.shadow_ratio()
    if ratio is None or shadow_length_m < 0:
        return None
    return shadow_length_m * ratio


def estimated_floors(height_m: float | None, floor_height_m: float) -> int | None:
    """Whole floors implied by a height, or None if height/scale is unknown."""
    if height_m is None or floor_height_m <= 0:
        return None
    floors = int(round(height_m / floor_height_m))
    return max(floors, 1)


# -- image measurement -----------------------------------------------------

def _direction_vector(angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return math.cos(radians), math.sin(radians)


def _shadow_facing_samples(
    polygon_working: np.ndarray, direction: tuple[float, float], spacing_px: float
) -> list[tuple[float, float]]:
    """Points along the polygon edges whose outward normal faces the shadow.

    We only launch rays from the side of the footprint that should be casting
    the visible shadow; the sunlit side is skipped.
    """
    dx, dy = direction
    samples: list[tuple[float, float]] = []
    count = len(polygon_working)
    # Signed area tells us the winding, so the outward normal is unambiguous.
    signed_area = 0.0
    for i in range(count):
        x0, y0 = polygon_working[i]
        x1, y1 = polygon_working[(i + 1) % count]
        signed_area += x0 * y1 - x1 * y0
    # With this shoelace convention and image y pointing down, a positive
    # signed area is a clockwise-on-screen ring whose outward edge normal is
    # (edge_y, -edge_x); a negative area flips it.
    outward_sign = 1.0 if signed_area > 0 else -1.0

    for i in range(count):
        x0, y0 = polygon_working[i]
        x1, y1 = polygon_working[(i + 1) % count]
        ex, ey = x1 - x0, y1 - y0
        length = math.hypot(ex, ey)
        if length < 1e-6:
            continue
        # Outward normal of this edge.
        nx, ny = outward_sign * ey / length, outward_sign * -ex / length
        if nx * dx + ny * dy <= 0.15:      # edge does not face the shadow
            continue
        steps = max(int(length / spacing_px), 1)
        for s in range(steps + 1):
            t = s / steps
            samples.append((x0 + t * ex, y0 + t * ey))
    return samples


def _march_ray(
    shadow: np.ndarray,
    start: tuple[float, float],
    direction: tuple[float, float],
    max_len_px: float,
    gap_tolerance_px: int,
    start_offset_px: float,
) -> float | None:
    """Length of shadow run from `start` along `direction`, or None if the ray
    begins on ground that is not in shadow (so the edge cast no visible shadow).
    """
    dx, dy = direction
    height, width = shadow.shape
    gap = 0
    last_shadow = 0.0
    began = False
    x = start[0] + dx * start_offset_px
    y = start[1] + dy * start_offset_px
    step = 0.0
    while step <= max_len_px:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < width and 0 <= yi < height):
            break
        if shadow[yi, xi] > 0.5:
            began = True
            last_shadow = step
            gap = 0
        elif began:
            gap += 1
            if gap > gap_tolerance_px:
                break
        else:
            # Not yet in shadow near the wall; allow a couple of pixels of
            # bright kerb before giving up on this ray entirely.
            if step > gap_tolerance_px + 1:
                return None
        x += dx
        y += dy
        step += 1.0
    # `last_shadow` is measured from the offset start; the shadow actually
    # spans from the wall, so add back the offset we skipped over the kerb.
    return last_shadow + start_offset_px if began else None


def measure_shadow_length_px(
    polygon_working: np.ndarray,
    shadow: np.ndarray,
    direction_angle_deg: float,
    params: dict,
) -> tuple[float | None, int, float]:
    """Robust shadow length (working px) for one footprint.

    Returns (length, ray_count, agreement). `length` is None when too few rays
    found any shadow at all, i.e. the building casts no measurable shadow in
    this image.
    """
    direction = _direction_vector(direction_angle_deg)
    spacing = float(params.get("ray_spacing_px", 6.0))
    max_len = float(params.get("max_shadow_px", 400.0))
    gap_tolerance = int(params.get("gap_tolerance_px", 3))
    start_offset = float(params.get("start_offset_px", 1.5))
    min_rays = int(params.get("min_rays", 3))

    samples = _shadow_facing_samples(polygon_working, direction, spacing)
    lengths: list[float] = []
    for point in samples:
        run = _march_ray(shadow, point, direction, max_len, gap_tolerance, start_offset)
        if run is not None and run > 0.0:
            lengths.append(run)

    if len(lengths) < min_rays:
        return None, len(lengths), 0.0

    array = np.array(lengths, dtype=np.float64)
    median = float(np.median(array))
    if median <= 0.0:
        return None, len(lengths), 0.0
    # Spread relative to the median, via the robust IQR. Tight agreement ->
    # confidence near 1; a scattered set of ray lengths -> near 0.
    q25, q75 = np.percentile(array, [25, 75])
    agreement = float(np.clip(1.0 - (q75 - q25) / (median + 1e-6), 0.0, 1.0))
    return median, len(lengths), agreement


# -- orchestration ---------------------------------------------------------

def estimate(
    buildings: list[Building],
    shadow: np.ndarray,
    shadow_direction,
    scale: ScaleInfo,
    sun: SunPosition,
    scale_factor: float,
    params: dict,
) -> list[HeightEstimate]:
    """Estimate every building's height in place and return the evidence.

    `shadow` is the working-resolution shadow cue; footprints are in source
    pixels, so they are scaled into working space by `scale_factor`.
    """
    floor_height = float(params.get("floor_height_m", 3.2))
    estimates: list[HeightEstimate] = []

    if not (scale.known and sun.known):
        reason = []
        if not scale.known:
            reason.append("scale unknown")
        if not sun.known:
            reason.append("sun elevation unknown")
        note = "height not estimated: " + " and ".join(reason)
        log.warning("HEIGHTS NOT ESTIMATED - %s", " and ".join(reason))
        for building in buildings:
            estimates.append(HeightEstimate(None, None, 0, 0.0, 0.0, "not_estimated", note))
        return estimates

    for building in buildings:
        if building.user_edited and building.estimated_height_m is not None:
            estimates.append(
                HeightEstimate(
                    building.estimated_height_m, None, 0, 0.0,
                    building.confidence.height or 0.0, "user_edited",
                    "height set by a human; not recomputed",
                )
            )
            continue

        polygon = np.array(building.footprint, dtype=np.float64) * scale_factor
        length_px_working, ray_count, agreement = measure_shadow_length_px(
            polygon, shadow, shadow_direction.angle_deg, params
        )

        if length_px_working is None:
            est = HeightEstimate(
                None, None, ray_count, agreement, 0.0, "not_estimated",
                "no measurable shadow on the expected side",
            )
            _apply(building, est, floor_height)
            estimates.append(est)
            continue

        length_px_source = length_px_working / scale_factor
        shadow_length_m = length_px_source * scale.meters_per_pixel
        height_m = height_from_shadow_m(shadow_length_m, sun)

        confidence = _confidence(agreement, shadow_direction.confidence, sun.confidence, ray_count)
        est = HeightEstimate(
            height_m=round(height_m, 2) if height_m is not None else None,
            shadow_length_px=round(length_px_source, 2),
            ray_count=ray_count,
            ray_agreement=round(agreement, 3),
            confidence=round(confidence, 3),
            source="shadow",
            note=f"{ray_count} rays, agreement {agreement:.2f}",
        )
        _apply(building, est, floor_height)
        estimates.append(est)

    measured = sum(1 for e in estimates if e.source == "shadow")
    log.info("estimated heights for %d/%d buildings from shadows", measured, len(buildings))
    return estimates


def _confidence(agreement: float, direction_conf: float, sun_conf: float, ray_count: int) -> float:
    """A height is only as trustworthy as its weakest input.

    Ray agreement, the shadow-direction estimate and the sun position all gate
    it multiplicatively - a good measurement in a wrongly-estimated direction
    is still wrong - and sparse rays cap the ceiling.
    """
    ray_factor = min(ray_count / 8.0, 1.0)
    base = agreement * max(direction_conf, 0.2) * max(sun_conf, 0.2)
    return float(np.clip(base * (0.5 + 0.5 * ray_factor), 0.0, 0.95))


def _apply(building: Building, est: HeightEstimate, floor_height: float) -> None:
    building.estimated_height_m = est.height_m
    building.height_source = est.source
    if est.height_m is not None:
        floors = estimated_floors(est.height_m, floor_height)
        building.estimated_floors = floors
        building.floors_source = "height_over_floor_height" if floors else "not_estimated"
        building.confidence.height = est.confidence
        building.confidence.floors = round(est.confidence * 0.8, 3)
        building.evidence = {
            **building.evidence,
            "shadow_length_px": est.shadow_length_px,
            "shadow_rays": est.ray_count,
            "ray_agreement": est.ray_agreement,
        }
