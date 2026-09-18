"""Stage 1 - preprocessing.

Mean-shift filtering flattens roof texture (gravel, HVAC units, parked cars)
while preserving the roof/ground boundary, which is what the later watershed
needs.  It is deterministic: no random seeds are involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Preprocessed:
    bgr: np.ndarray          # mean-shift smoothed BGR
    lab: np.ndarray          # LAB of the smoothed image, float32
    lightness: np.ndarray    # L channel of the smoothed image, float32
    source_lab: np.ndarray   # LAB of the unsmoothed image, float32


def run(working_bgr: np.ndarray, params: dict) -> Preprocessed:
    sp = int(params.get("meanshift_spatial_radius", 9))
    sr = int(params.get("meanshift_color_radius", 18))
    levels = int(params.get("meanshift_pyramid_levels", 2))
    log.info("mean-shift smoothing (sp=%d, sr=%d, levels=%d)", sp, sr, levels)
    smoothed = cv2.pyrMeanShiftFiltering(working_bgr, sp=sp, sr=sr, maxLevel=levels)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return Preprocessed(
        bgr=smoothed, lab=lab, lightness=lab[:, :, 0].copy(), source_lab=source_lab
    )
