"""Stage 3 - shadow direction estimation.

In a single nadir image the most reliable evidence that a bright blob is a
raised structure is a shadow attached to it on a consistent side.  We recover
that direction by finding the in-plane offset that best aligns the shadow mask
with bright surfaces.

The result is also the input Phase 2 needs for shadow-based height estimation,
which is why it is a stage of its own rather than a detail of the detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ShadowDirection:
    """Direction, in image frame degrees, pointing from a structure to its shadow.

    `confidence` reflects how peaked the alignment score was; a flat response
    (few or ambiguous shadows) yields a low value and callers should down-weight
    the shadow cue accordingly.
    """

    angle_deg: float
    offset_px: float
    score: float
    confidence: float
    source: str   # "estimated" | "user_override"


def estimate(cues, params: dict) -> ShadowDirection:
    override = params.get("override_deg")
    if override is not None:
        log.info("shadow direction fixed by config: %.1f deg", float(override))
        return ShadowDirection(float(override), 20.0, 1.0, 1.0, "user_override")

    height, width = cues.shadow.shape
    shadow_blur = cv2.GaussianBlur(cues.shadow, (0, 0), 4.0)
    bright = (cues.lightness > np.percentile(cues.lightness, 80.0)).astype(np.float32)
    bright_sum = float(bright.sum()) + 1e-6

    step = int(params.get("search_step_deg", 5))
    distances = [float(d) for d in params.get("search_distances_px", [8, 14, 20, 28])]

    results: list[tuple[float, float, float]] = []   # score, angle, distance
    for angle in range(0, 360, step):
        radians = np.deg2rad(angle)
        best_for_angle = 0.0
        best_distance = distances[0]
        for distance in distances:
            # Shift the shadow map *against* the candidate direction; if the
            # direction is right, shadows land on the structures casting them.
            matrix = np.float32(
                [[1, 0, -distance * np.cos(radians)], [0, 1, -distance * np.sin(radians)]]
            )
            shifted = cv2.warpAffine(shadow_blur, matrix, (width, height))
            score = float((bright * shifted).sum() / bright_sum)
            if score > best_for_angle:
                best_for_angle, best_distance = score, distance
        results.append((best_for_angle, float(angle), best_distance))

    results.sort(reverse=True)
    best_score, best_angle, best_distance = results[0]
    scores = np.array([r[0] for r in results], dtype=np.float64)
    median = float(np.median(scores))
    # Peakedness: how far the best response stands above the typical response.
    peak_ratio = (best_score - median) / (best_score + 1e-9) if best_score > 0 else 0.0
    confidence = float(np.clip(peak_ratio * 1.6, 0.0, 0.95))

    log.info(
        "shadow direction %.0f deg at %.0f px (score %.4f, confidence %.2f)",
        best_angle, best_distance, best_score, confidence,
    )
    return ShadowDirection(best_angle, best_distance, best_score, confidence, "estimated")


def adjacency_map(cues, direction: ShadowDirection) -> np.ndarray:
    """Map of 'a shadow lies on the expected side of this pixel', in [0, 1]."""
    height, width = cues.shadow.shape
    radians = np.deg2rad(direction.angle_deg)
    distance = max(direction.offset_px, 6.0)
    blurred = cv2.GaussianBlur(cues.shadow, (0, 0), 5.0)
    matrix = np.float32(
        [[1, 0, -distance * np.cos(radians)], [0, 1, -distance * np.sin(radians)]]
    )
    shifted = cv2.warpAffine(blurred, matrix, (width, height))
    peak = float(shifted.max())
    return shifted / peak if peak > 0 else shifted
