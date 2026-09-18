"""Street network generation by recursive subdivision.

A site rectangle is split in two by a road, then each half is split again,
until the remaining blocks are small enough to build on. Splitting the longer
axis keeps blocks from becoming slivers, and jittering the split position stops
the result looking like graph paper.

Road width falls with recursion depth, which produces a natural hierarchy:
a few wide through-roads, then feeder streets, then residential lanes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rect:
    """Axis-aligned rectangle in metres."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def inset(self, amount: float) -> "Rect | None":
        """Shrink on all sides; None if nothing survives."""
        w = self.w - 2 * amount
        h = self.h - 2 * amount
        if w <= 0.5 or h <= 0.5:
            return None
        return Rect(self.x + amount, self.y + amount, w, h)

    def corners(self) -> list[tuple[float, float]]:
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]


@dataclass
class Road:
    """A road as a rectangular strip, with the hierarchy level that made it."""

    rect: Rect
    level: int
    axis: str   # "v" = the road runs vertically, "h" = horizontally


@dataclass
class Network:
    blocks: list[Rect]
    roads: list[Road]


def generate(site: Rect, params: dict, rng: np.random.Generator) -> Network:
    target_area = float(params.get("target_block_area_m2", 5200.0))
    min_side = float(params.get("min_block_side_m", 42.0))
    max_depth = int(params.get("max_depth", 6))
    widths = [float(w) for w in params.get("road_widths_m", [18.0, 12.0, 9.0, 7.0])]
    jitter = float(params.get("split_jitter", 0.12))

    blocks: list[Rect] = []
    roads: list[Road] = []

    def width_for(depth: int) -> float:
        return widths[min(depth, len(widths) - 1)]

    def split(rect: Rect, depth: int) -> None:
        road_width = width_for(depth)
        # Stop when another split would leave unusably small blocks.
        too_small = rect.area <= target_area or depth >= max_depth
        can_split_v = rect.w >= 2 * min_side + road_width
        can_split_h = rect.h >= 2 * min_side + road_width
        if too_small or not (can_split_v or can_split_h):
            blocks.append(rect)
            return

        axis = "v" if (rect.w >= rect.h and can_split_v) or not can_split_h else "h"
        span = rect.w if axis == "v" else rect.h
        usable = span - road_width
        fraction = 0.5 + float(rng.uniform(-jitter, jitter))
        first = usable * fraction
        # Keep both sides buildable.
        first = float(np.clip(first, min_side, usable - min_side))
        second = usable - first

        if axis == "v":
            left = Rect(rect.x, rect.y, first, rect.h)
            road = Rect(rect.x + first, rect.y, road_width, rect.h)
            right = Rect(road.x + road_width, rect.y, second, rect.h)
            roads.append(Road(road, depth, "v"))
            split(left, depth + 1)
            split(right, depth + 1)
        else:
            bottom = Rect(rect.x, rect.y, rect.w, first)
            road = Rect(rect.x, rect.y + first, rect.w, road_width)
            top = Rect(rect.x, road.y + road_width, rect.w, second)
            roads.append(Road(road, depth, "h"))
            split(bottom, depth + 1)
            split(top, depth + 1)

    split(site, 0)
    return Network(blocks=blocks, roads=roads)
