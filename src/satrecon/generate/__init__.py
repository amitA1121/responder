"""Procedural scene generation.

The counterpart to `detect/`: instead of recovering buildings from an image,
these modules synthesise them from parameters. Both paths produce the same
`Scene`, so meshing, export and the review page are shared.
"""

from .base import SceneGenerator, available_generators, get_generator, register_generator

__all__ = [
    "SceneGenerator",
    "get_generator",
    "register_generator",
    "available_generators",
]
