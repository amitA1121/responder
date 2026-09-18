"""Stage 2 - per-pixel surface cues.

These are deliberately simple, explainable masks rather than a black box.
Each one answers a single question about a pixel, and the detector combines
them with weights that live in the config file.

None of these cues identify *what* a structure is; they only separate likely
roof surfaces from vegetation, shadow and bare ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class CueMaps:
    vegetation: np.ndarray   # float32 {0,1}
    shadow: np.ndarray       # float32 {0,1}
    soil: np.ndarray         # float32 {0,1} - bare/disturbed ground
    texture: np.ndarray      # float32, local std of lightness
    lightness: np.ndarray    # float32, unsmoothed L
    bright_reference: float  # L anchor where a surface starts counting as bright
    bright_high: float       # L at which the brightness cue saturates


def run(working_bgr: np.ndarray, source_lab: np.ndarray, params: dict) -> CueMaps:
    blue, green, red = (c.astype(np.float32) for c in cv2.split(working_bgr))
    lightness = source_lab[:, :, 0]
    a_chan = source_lab[:, :, 1]
    b_chan = source_lab[:, :, 2]

    # Excess green: a standard, sensor-agnostic greenness index for RGB.
    excess_green = 2.0 * green - red - blue
    vegetation = (
        (excess_green > float(params.get("vegetation_exg_threshold", 8.0)))
        & (lightness < float(params.get("vegetation_max_lightness", 175.0)))
    ).astype(np.float32)

    shadow_cut = float(np.percentile(lightness, float(params.get("shadow_lightness_percentile", 14.0))))
    shadow = (lightness < shadow_cut).astype(np.float32)

    chroma = np.sqrt((a_chan - 128.0) ** 2 + (b_chan - 128.0) ** 2)
    soil = (
        (chroma > float(params.get("soil_min_chroma", 9.0)))
        & (a_chan > float(params.get("soil_min_a", 132.0)))
    ).astype(np.float32)

    window = int(params.get("texture_window_px", 11)) | 1
    mean = cv2.blur(lightness, (window, window))
    mean_sq = cv2.blur(lightness * lightness, (window, window))
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    # Data-adaptive brightness anchors: a scene of pale concrete and a scene
    # of dark roofs both get a usable 0..1 ramp.
    bright_reference = float(np.percentile(lightness, 50.0))
    bright_high = float(np.percentile(lightness, 93.0))
    log.info(
        "cues: vegetation %.1f%%, shadow %.1f%% (L<%.0f), bare ground %.1f%%",
        100 * vegetation.mean(), 100 * shadow.mean(), shadow_cut, 100 * soil.mean(),
    )
    return CueMaps(
        vegetation=vegetation,
        shadow=shadow,
        soil=soil,
        texture=texture,
        lightness=lightness,
        bright_reference=bright_reference,
        bright_high=bright_high,
    )
