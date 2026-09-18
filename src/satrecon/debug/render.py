"""Stage 7 - 2D debug renders.

The point of Phase 1 is visual verification, so every intermediate raster the
detector relied on is written out next to the final overlay.  If a footprint
looks wrong, these images show which cue caused it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..model import Scene

log = get_logger(__name__)

BAND_COLORS = {          # BGR
    "HIGH": (90, 220, 90),
    "MEDIUM": (60, 200, 255),
    "LOW": (80, 80, 245),
}


def _to_uint8(layer: np.ndarray) -> np.ndarray:
    if layer.dtype == np.uint8:
        return layer
    if layer.ndim == 3:
        return np.clip(layer, 0, 255).astype(np.uint8)
    finite = layer[np.isfinite(layer)]
    if finite.size == 0:
        return np.zeros(layer.shape, np.uint8)
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-9:
        return np.zeros(layer.shape, np.uint8)
    return np.clip((layer - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-colouring of a label image."""
    lut = np.zeros((int(labels.max()) + 2, 3), np.uint8)
    rng = np.random.default_rng(12345)          # fixed seed: reproducible
    lut[1:] = rng.integers(40, 235, size=(lut.shape[0] - 1, 3), dtype=np.uint8)
    safe = np.clip(labels, 0, lut.shape[0] - 1)
    return lut[safe]


def write_layers(layers: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [
        "preprocessed", "vegetation", "shadow", "soil",
        "shadow_adjacency", "gradient", "segments", "region_score", "accepted_mask",
    ]
    written: list[Path] = []
    for index, name in enumerate([n for n in order if n in layers], start=1):
        layer = layers[name]
        image = _colorize_labels(layer) if name == "segments" else _to_uint8(layer)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        path = out_dir / f"{index:02d}_{name}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    return written


def render_overlay(scene: Scene, source_bgr: np.ndarray, out_path: Path, label: bool = True) -> Path:
    """Footprints drawn over the original image, coloured by confidence band."""
    canvas = source_bgr.copy()
    fill = canvas.copy()
    for building in scene.buildings:
        polygon = np.array(building.footprint, dtype=np.int32)
        color = BAND_COLORS[building.confidence.band()]
        cv2.fillPoly(fill, [polygon], color)
    cv2.addWeighted(fill, 0.22, canvas, 0.78, 0, canvas)

    for building in scene.buildings:
        polygon = np.array(building.footprint, dtype=np.int32)
        color = BAND_COLORS[building.confidence.band()]
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        if label:
            x, y, w, h = building.bounding_box
            text = building.id
            origin = (int(x + w / 2 - 16), int(y + h / 2))
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    _draw_legend(canvas, scene)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    log.info("wrote overlay %s", out_path)
    return out_path


def _draw_legend(canvas: np.ndarray, scene: Scene) -> None:
    lines = [
        f"{len(scene.buildings)} footprint candidates",
        "scale: " + (
            f"{scene.scale.meters_per_pixel:.4f} m/px ({scene.scale.source})"
            if scene.scale.known else "UNKNOWN - relative estimates"
        ),
        "confidence: green HIGH / amber MEDIUM / red LOW",
        "approximate - not a measurement",
    ]
    pad, line_height = 10, 22
    box_height = line_height * len(lines) + pad
    cv2.rectangle(canvas, (0, 0), (430, box_height), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (pad, pad + line_height * index + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA,
        )


def render_footprints_only(scene: Scene, size: tuple[int, int], out_path: Path) -> Path:
    """Footprints on a neutral background - useful for checking geometry alone."""
    width, height = size
    canvas = np.full((height, width, 3), 18, np.uint8)
    for building in scene.buildings:
        polygon = np.array(building.footprint, dtype=np.int32)
        color = BAND_COLORS[building.confidence.band()]
        cv2.fillPoly(canvas, [polygon], tuple(int(c * 0.35) for c in color))
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path
