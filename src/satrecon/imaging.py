"""Image loading, hashing and working-resolution management."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class LoadedImage:
    """Source image plus the (possibly downscaled) array the pipeline works on.

    `scale_factor` is working_size / source_size.  All geometry produced at
    working resolution is mapped back to source pixels before being stored, so
    the persisted data model always speaks source-image coordinates.
    """

    path: Path
    sha256: str
    source_bgr: np.ndarray
    working_bgr: np.ndarray
    scale_factor: float

    @property
    def source_size(self) -> tuple[int, int]:
        h, w = self.source_bgr.shape[:2]
        return w, h

    def to_source_coords(self, points: np.ndarray) -> np.ndarray:
        if self.scale_factor == 1.0:
            return points.astype(np.float64)
        return points.astype(np.float64) / self.scale_factor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: str | Path, max_working_edge: int = 2000) -> LoadedImage:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported image type '{p.suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {p}")

    height, width = image.shape[:2]
    longest = max(width, height)
    if longest > max_working_edge:
        factor = max_working_edge / longest
        working = cv2.resize(
            image,
            (int(round(width * factor)), int(round(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
        log.info(
            "image %dx%d downscaled to %dx%d for processing (factor %.4f)",
            width, height, working.shape[1], working.shape[0], factor,
        )
    else:
        factor = 1.0
        working = image

    digest = sha256_file(p)
    log.info("loaded %s (%dx%d, sha256=%s...)", p.name, width, height, digest[:12])
    return LoadedImage(
        path=p, sha256=digest, source_bgr=image, working_bgr=working, scale_factor=factor
    )
