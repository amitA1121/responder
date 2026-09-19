"""Detector interface.

Any detector - classical CV, an ONNX segmentation model, a remote vision API,
or a file of hand-drawn polygons - implements `detect()` and returns
`BuildingDetection` objects in working-image pixel coordinates.  Nothing
downstream knows which detector produced them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..model import BuildingDetection


@dataclass
class DetectionContext:
    """Shared, already-computed inputs offered to a detector.

    A detector may ignore any of these and work from `bgr` alone; they are
    provided so that stages are not recomputed per detector.
    """

    bgr: np.ndarray
    scale_factor: float = 1.0
    preprocessed: Any = None
    cues: Any = None
    shadow_direction: Any = None
    shadow_adjacency: np.ndarray | None = None
    segmentation: Any = None


@dataclass
class DetectorResult:
    detections: list[BuildingDetection]
    # Intermediate rasters a debug view may render; keys are free-form.
    debug_layers: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class BuildingDetector(ABC):
    """Contract: detect(image) -> BuildingDetection[]."""

    name: str = "abstract"

    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def detect(self, context: DetectionContext) -> DetectorResult:
        """Return building candidates in working-image pixel coordinates."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}
