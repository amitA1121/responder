"""Stage 4b - per-region descriptors.

Kept apart from the detector so that an alternative detector (or a future
learned model) can reuse exactly the same descriptors, and so each feature can
be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class RegionFeatures:
    label: int
    area_px: int
    thickness_px: float          # max inscribed radius - roads are thin, roofs are not
    rectangularity: float        # area / min-area-rect area
    elongation: float            # min-area-rect long side / short side
    solidity: float              # area / convex hull area
    mean_lightness: float
    mean_texture: float
    vegetation_fraction: float
    shadow_fraction: float
    soil_fraction: float
    shadow_adjacency: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    orientation_deg: float
    rect_size: tuple[float, float]
    extras: dict[str, Any] = field(default_factory=dict)


def _region_means(labels: np.ndarray, count: int, *maps: np.ndarray) -> list[np.ndarray]:
    flat = labels.ravel().clip(0)
    sizes = np.bincount(flat, minlength=count).astype(np.float64)
    sizes[sizes == 0] = 1.0
    return [np.bincount(flat, weights=m.ravel(), minlength=count) / sizes for m in maps]


def compute(
    segmentation, cues, adjacency: np.ndarray, min_area_px: int
) -> list[RegionFeatures]:
    labels = segmentation.labels
    count = int(labels.max()) + 1
    mean_light, mean_texture, veg_frac, shadow_frac, soil_frac, adj = _region_means(
        labels, count, cues.lightness, cues.texture,
        cues.vegetation, cues.shadow, cues.soil, adjacency,
    )

    areas = np.bincount(labels.ravel().clip(0), minlength=count)
    features: list[RegionFeatures] = []
    for label in range(1, count):
        area = int(areas[label])
        if area < min_area_px:
            continue
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 1.0:
            continue

        # Padding keeps the distance transform honest for regions touching the
        # image border, which would otherwise look artificially thick.
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        thickness = float(cv2.distanceTransform(padded, cv2.DIST_L2, 5).max())

        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        rect_area = max(rw * rh, 1.0)
        hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
        x, y, w, h = cv2.boundingRect(contour)

        # Report the long axis, measured counter-clockwise from +x.
        if rw >= rh:
            long_side, short_side, orientation = rw, rh, -angle
        else:
            long_side, short_side, orientation = rh, rw, -(angle + 90.0)

        features.append(
            RegionFeatures(
                label=label,
                area_px=area,
                thickness_px=thickness,
                rectangularity=float(area / rect_area),
                elongation=float(long_side / max(short_side, 1e-3)),
                solidity=float(area / hull_area),
                mean_lightness=float(mean_light[label]),
                mean_texture=float(mean_texture[label]),
                vegetation_fraction=float(veg_frac[label]),
                shadow_fraction=float(shadow_frac[label]),
                soil_fraction=float(soil_frac[label]),
                shadow_adjacency=float(adj[label]),
                centroid=(float(cx), float(cy)),
                bounding_box=(int(x), int(y), int(w), int(h)),
                orientation_deg=float(orientation % 180.0),
                rect_size=(float(long_side), float(short_side)),
            )
        )
    return features
