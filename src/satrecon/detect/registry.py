"""Detector registry - lets a new detector be plugged in by name."""

from __future__ import annotations

from typing import Callable

from .base import BuildingDetector

_REGISTRY: dict[str, Callable[..., BuildingDetector]] = {}


def register_detector(name: str, factory: Callable[..., BuildingDetector]) -> None:
    _REGISTRY[name] = factory


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def get_detector(name: str, params: dict | None = None) -> BuildingDetector:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown detector '{name}'. Available: {', '.join(available_detectors()) or 'none'}"
        )
    return _REGISTRY[name](params or {})
