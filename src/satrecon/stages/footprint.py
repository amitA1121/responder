"""Stage 5 - footprint extraction.

Turns a detected contour into a clean polygon: noise removed, redundant
vertices dropped, meaningful corners preserved.  Irregular (L-, U-, cross-
shaped) outlines are supported - no rectangle is ever forced onto a detection.

The dense contour is retained alongside the simplified polygon so that a
later manual correction can fall back to the original evidence.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..model import Building, BuildingDetection, Confidence, ScaleInfo

log = get_logger(__name__)


def _simplify(contour: np.ndarray, params: dict) -> np.ndarray:
    fraction = float(params.get("simplify_fraction", 0.012))
    min_vertices = int(params.get("min_vertices", 4))
    max_vertices = int(params.get("max_vertices", 40))
    perimeter = cv2.arcLength(contour, True)

    simplified = cv2.approxPolyDP(contour, fraction * perimeter, True)
    # Too coarse: relax until meaningful corners survive.
    scale = fraction
    while len(simplified) < min_vertices and scale > 1e-4:
        scale *= 0.5
        simplified = cv2.approxPolyDP(contour, scale * perimeter, True)
    # Too fine: tighten until the vertex budget is met.
    scale = fraction
    while len(simplified) > max_vertices and scale < 0.25:
        scale *= 1.5
        simplified = cv2.approxPolyDP(contour, scale * perimeter, True)
    return simplified.reshape(-1, 2).astype(np.float64)


def _regularize(polygon: np.ndarray, angle_deg: float, tolerance_deg: float) -> np.ndarray:
    """Snap near-axis edges onto the building's dominant axis.

    Approximate by design: it tidies jagged watershed boundaries without
    forcing the outline to be rectangular. Vertices move by at most the
    original edge deviation.
    """
    if len(polygon) < 4:
        return polygon
    theta = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    local = polygon @ rotation.T
    count = len(local)
    for i in range(count):
        j = (i + 1) % count
        dx, dy = local[j] - local[i]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        edge_angle = abs(np.rad2deg(np.arctan2(dy, dx))) % 180.0
        if edge_angle <= tolerance_deg or edge_angle >= 180.0 - tolerance_deg:
            mean_y = (local[i, 1] + local[j, 1]) / 2.0
            local[i, 1] = local[j, 1] = mean_y
        elif abs(edge_angle - 90.0) <= tolerance_deg:
            mean_x = (local[i, 0] + local[j, 0]) / 2.0
            local[i, 0] = local[j, 0] = mean_x
    return local @ rotation


def _polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _footprint_confidence(detection: BuildingDetection, polygon: np.ndarray) -> float:
    """Combine detector confidence with how faithfully the polygon fits.

    A simplified polygon that departs badly from the detected contour is a
    sign the outline is unreliable, regardless of the detector's own score.
    """
    contour_area = max(detection.area_px, 1.0)
    fit = 1.0 - min(abs(_polygon_area(polygon) - contour_area) / contour_area, 1.0)
    return float(np.clip(0.65 * detection.confidence + 0.35 * fit, 0.0, 1.0))


def build(
    detections: list[BuildingDetection],
    scale: ScaleInfo,
    params: dict,
    to_source,
) -> list[Building]:
    """Convert detections into `Building` records in source-image pixels."""
    min_area = float(params.get("min_area_px", 900))
    regularize = bool(params.get("regularize", True))
    tolerance = float(params.get("regularize_angle_tolerance_deg", 12.0))

    buildings: list[Building] = []
    for detection in detections:
        contour = np.array(detection.contour, dtype=np.float32).reshape(-1, 1, 2)
        polygon = _simplify(contour, params)
        if len(polygon) < 3:
            continue
        if regularize:
            polygon = _regularize(polygon, detection.orientation_deg, tolerance)

        source_polygon = to_source(polygon)
        source_contour = to_source(np.array(detection.contour, dtype=np.float64))
        area_px = _polygon_area(source_polygon)
        if area_px < min_area:
            continue

        rect = cv2.minAreaRect(source_polygon.astype(np.float32))
        (_, _), (rw, rh), angle = rect
        length_px, width_px = (rw, rh) if rw >= rh else (rh, rw)
        orientation = (-angle if rw >= rh else -(angle + 90.0)) % 180.0
        x, y, w, h = cv2.boundingRect(source_polygon.astype(np.float32))

        confidence = Confidence(
            footprint=round(_footprint_confidence(detection, polygon), 4),
            # An unknown scale is "not estimated", not "estimated badly"; it is
            # surfaced separately rather than folded into the footprint band.
            scale=round(scale.confidence, 4) if scale.known else None,
            height=None,   # Phase 2
            floors=None,   # Phase 2
        )

        buildings.append(
            Building(
                id=detection.id.replace("D", "B"),
                footprint=[(round(float(px), 2), round(float(py), 2)) for px, py in source_polygon],
                contour=[(round(float(px), 1), round(float(py), 1)) for px, py in source_contour],
                bounding_box=(int(x), int(y), int(w), int(h)),
                orientation_deg=round(float(orientation), 3),
                area_px=round(area_px, 2),
                length_px=round(float(length_px), 2),
                width_px=round(float(width_px), 2),
                length_m=_round_opt(scale.px_to_m(length_px)),
                width_m=_round_opt(scale.px_to_m(width_px)),
                area_m2=_round_opt(
                    area_px * scale.meters_per_pixel ** 2 if scale.known else None
                ),
                confidence=confidence,
                evidence=detection.evidence,
            )
        )

    # Stable, deterministic ordering: largest first, ties broken by position.
    buildings.sort(key=lambda b: (-b.area_px, b.bounding_box[1], b.bounding_box[0]))
    for index, building in enumerate(buildings, start=1):
        building.id = f"B{index:03d}"
    log.info("extracted %d footprints (min area %.0f px)", len(buildings), min_area)
    return buildings


def _round_opt(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)
