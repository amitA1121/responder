"""Lot subdivision within a block.

Blocks are cut into plots by the same recursive split used for roads, but
without a gap between them - neighbouring plots share a boundary. A plot is
then set back from its edges to leave the yard/verge that makes a neighbourhood
read as residential rather than as one continuous slab.
"""

from __future__ import annotations

import numpy as np

from .roads import Rect


def subdivide(block: Rect, params: dict, rng: np.random.Generator) -> list[Rect]:
    target_area = float(params.get("target_lot_area_m2", 900.0))
    min_side = float(params.get("min_lot_side_m", 14.0))
    jitter = float(params.get("split_jitter", 0.16))
    max_depth = int(params.get("max_depth", 5))

    lots: list[Rect] = []

    def split(rect: Rect, depth: int) -> None:
        can_v = rect.w >= 2 * min_side
        can_h = rect.h >= 2 * min_side
        if rect.area <= target_area or depth >= max_depth or not (can_v or can_h):
            lots.append(rect)
            return
        axis = "v" if (rect.w >= rect.h and can_v) or not can_h else "h"
        span = rect.w if axis == "v" else rect.h
        fraction = 0.5 + float(rng.uniform(-jitter, jitter))
        first = float(np.clip(span * fraction, min_side, span - min_side))
        if axis == "v":
            split(Rect(rect.x, rect.y, first, rect.h), depth + 1)
            split(Rect(rect.x + first, rect.y, span - first, rect.h), depth + 1)
        else:
            split(Rect(rect.x, rect.y, rect.w, first), depth + 1)
            split(Rect(rect.x, rect.y + first, rect.w, span - first), depth + 1)

    split(block, 0)
    return lots
