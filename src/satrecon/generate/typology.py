"""Building typologies.

A typology bundles the things that co-vary in real housing stock: footprint
size, floor count, floor height and roof form. Keeping them together stops the
generator producing a two-storey cottage with a 40 m footprint.

Everything here is data-driven from config, so a different housing pattern is
a config change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roads import Rect

DEFAULT_TYPOLOGIES: dict[str, dict] = {
    "tower": {
        "weight": 1.0,
        "floors": [12, 24],
        "floor_height_m": 3.1,
        "max_footprint_m": [34.0, 34.0],
        "setback_m": 7.0,
        "roof": "flat",
        "color": [0.82, 0.80, 0.76],
    },
    "slab": {
        "weight": 2.0,
        "floors": [5, 9],
        "floor_height_m": 3.0,
        "max_footprint_m": [48.0, 18.0],
        "setback_m": 5.0,
        "roof": "flat",
        "color": [0.78, 0.76, 0.71],
    },
    "rowhouse": {
        "weight": 2.5,
        "floors": [2, 4],
        "floor_height_m": 3.0,
        "max_footprint_m": [30.0, 12.0],
        "setback_m": 3.5,
        "roof": "gabled",
        "color": [0.74, 0.66, 0.58],
    },
    "detached": {
        "weight": 2.0,
        "floors": [1, 2],
        "floor_height_m": 3.0,
        "max_footprint_m": [16.0, 13.0],
        "setback_m": 3.0,
        "roof": "hip",
        "color": [0.80, 0.72, 0.62],
    },
}


@dataclass
class Placement:
    """A concrete building specified on a lot."""

    typology: str
    footprint: Rect
    floors: int
    floor_height_m: float
    height_m: float
    roof_type: str
    color: tuple[float, float, float]


def _pick(names: list[str], weights: np.ndarray, rng: np.random.Generator) -> str:
    return str(rng.choice(names, p=weights / weights.sum()))


def place(
    lot: Rect,
    typologies: dict[str, dict],
    density: float,
    rng: np.random.Generator,
) -> Placement | None:
    """Choose a typology that fits this lot and size a building inside it.

    `density` in [0, 1] biases selection toward taller types - use it to make a
    centre denser than the edges.
    """
    names, weights = [], []
    for name, spec in typologies.items():
        max_w, max_h = spec["max_footprint_m"]
        setback = spec["setback_m"]
        # The lot must hold the smallest useful version of this type.
        if lot.w < min(max_w, max_h) * 0.45 + 2 * setback:
            continue
        if lot.h < min(max_w, max_h) * 0.45 + 2 * setback:
            continue
        floors_mid = (spec["floors"][0] + spec["floors"][1]) / 2.0
        # Tall types gain weight with density, low types lose it.
        tilt = 1.0 + density * (floors_mid / 8.0 - 1.0)
        names.append(name)
        weights.append(max(float(spec["weight"]) * max(tilt, 0.05), 1e-4))

    if not names:
        return None

    name = _pick(names, np.array(weights, dtype=np.float64), rng)
    spec = typologies[name]
    setback = float(spec["setback_m"])
    buildable = lot.inset(setback)
    if buildable is None:
        return None

    max_w, max_h = (float(v) for v in spec["max_footprint_m"])
    # Orient the long side of the type along the long side of the lot.
    if (buildable.w >= buildable.h) != (max_w >= max_h):
        max_w, max_h = max_h, max_w

    width = min(buildable.w, max_w * float(rng.uniform(0.72, 1.0)))
    depth = min(buildable.h, max_h * float(rng.uniform(0.72, 1.0)))
    if width < 4.0 or depth < 4.0:
        return None

    # Centre it in the buildable area, with a little slack for variety.
    slack_x = max(buildable.w - width, 0.0)
    slack_y = max(buildable.h - depth, 0.0)
    x = buildable.x + slack_x * float(rng.uniform(0.3, 0.7))
    y = buildable.y + slack_y * float(rng.uniform(0.3, 0.7))

    low, high = spec["floors"]
    floors = int(rng.integers(int(low), int(high) + 1))
    floor_height = float(spec["floor_height_m"])
    return Placement(
        typology=name,
        footprint=Rect(x, y, width, depth),
        floors=floors,
        floor_height_m=floor_height,
        height_m=round(floors * floor_height, 3),
        roof_type=str(spec["roof"]),
        color=tuple(float(c) for c in spec["color"]),
    )
