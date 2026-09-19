from .base import BuildingDetector, DetectionContext, DetectorResult
from .registry import get_detector, register_detector, available_detectors

# Import built-in detectors so the registry is populated immediately after the
# package is imported, and so the CLI can list all detectors without extra work.
from . import classical  # noqa: F401
from . import manual  # noqa: F401

__all__ = [
    "BuildingDetector",
    "DetectionContext",
    "DetectorResult",
    "get_detector",
    "register_detector",
    "available_detectors",
]
