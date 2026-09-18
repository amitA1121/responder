"""Stage 4 - edge-aware region segmentation.

Global colour thresholds cannot separate roofs from roads: both are bright,
low-chroma surfaces.  What does separate them is the *boundary* - roofs are
bounded by strong, closed, straight edges.  So the image is partitioned into
regions whose borders follow those edges, and classification happens per region
rather than per pixel.

Seeds are the low-gradient plateaus of the smoothed image, so the region count
adapts to scene complexity instead of being fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Segmentation:
    labels: np.ndarray       # int32, watershed labels (-1 on boundaries)
    gradient: np.ndarray     # uint8, normalised gradient magnitude
    region_count: int


def run(pre, params: dict) -> Segmentation:
    lightness = pre.lightness
    grad_x = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), float(params.get("gradient_blur_sigma", 1.2)))
    reference = float(np.percentile(magnitude, 99.0)) or 1.0
    gradient = np.clip(magnitude / reference * 255.0, 0, 255).astype(np.uint8)

    seed_cut = float(params.get("gradient_seed_percentile", 18.0))
    seeds = (gradient < seed_cut).astype(np.uint8)
    kernel_size = int(params.get("seed_open_kernel_px", 7)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, kernel, iterations=1)

    count, markers = cv2.connectedComponents(seeds)
    if count <= 1:
        raise RuntimeError(
            "segmentation found no low-gradient seed regions; "
            "try raising segmentation.gradient_seed_percentile"
        )
    labels = cv2.watershed(pre.bgr, markers.astype(np.int32))
    log.info("segmented into %d regions", count - 1)
    return Segmentation(labels=labels, gradient=gradient, region_count=count - 1)
