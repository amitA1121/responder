"""Generator interface and registry.

Mirrors `detect/base.py` deliberately: a generator is swappable the same way a
detector is, so a hand-authored layout or a different urban model can replace
the built-in one without touching anything downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..model import Scene


class SceneGenerator(ABC):
    """Contract: generate(seed) -> Scene."""

    name: str = "abstract"

    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def generate(self, seed: int) -> Scene:
        """Produce a complete scene. Identical seed must give identical output."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}


_REGISTRY: dict[str, Callable[..., SceneGenerator]] = {}


def register_generator(name: str, factory: Callable[..., SceneGenerator]) -> None:
    _REGISTRY[name] = factory


def available_generators() -> list[str]:
    return sorted(_REGISTRY)


def get_generator(name: str, params: dict | None = None) -> SceneGenerator:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown generator '{name}'. Available: {', '.join(available_generators()) or 'none'}"
        )
    return _REGISTRY[name](params or {})
