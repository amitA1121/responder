"""Exporter interface.

Same shape as the detector and generator registries so a new target format is
an added module, not a change to the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ..model import Scene


class ThreeDExporter(ABC):
    name: str = "abstract"
    extension: str = ".bin"

    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def export(self, scene: Scene, out_path: Path) -> Path:
        """Write the scene and return the path actually written."""


_REGISTRY: dict[str, Callable[..., ThreeDExporter]] = {}


def register_exporter(name: str, factory: Callable[..., ThreeDExporter]) -> None:
    _REGISTRY[name] = factory


def available_exporters() -> list[str]:
    return sorted(_REGISTRY)


def get_exporter(name: str, params: dict | None = None) -> ThreeDExporter:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown exporter '{name}'. Available: {', '.join(available_exporters()) or 'none'}"
        )
    return _REGISTRY[name](params or {})
