"""Configuration loading.

Defaults live in code so the pipeline runs with no config file at all.  A YAML
file supplies overrides and, importantly, any *known* facts about the image
(scale, north direction, coordinates).  User-supplied values are never
overwritten by heuristics - see `scale.py` for how that is enforced.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .logging_setup import get_logger

log = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "image": {
        # Anticlockwise rotation, in degrees, from image-up to true north.
        # null => unknown; orientations are then reported image-relative.
        "north_offset_deg": None,
    },
    "scale": {
        # Authoritative if set by the user. null => derive or leave unknown.
        "meters_per_pixel": None,
        # Two image points with a known real-world separation.
        "reference": {
            "point_a": None,      # [x, y] in pixels
            "point_b": None,      # [x, y] in pixels
            "distance_meters": None,
        },
        # Ground sample distance in m/px, e.g. from a satellite product sheet.
        "ground_sample_distance": None,
    },
    "geo": {
        "center_lat": None,
        "center_lon": None,
        "crs": None,
    },
    "preprocess": {
        # Edge-preserving mean-shift smoothing. Flattens roof texture while
        # keeping roof/ground boundaries sharp.
        "meanshift_spatial_radius": 9,
        "meanshift_color_radius": 18,
        "meanshift_pyramid_levels": 2,
        # Images larger than this on the long edge are processed downscaled and
        # results are mapped back, keeping runtime predictable.
        "max_working_edge_px": 2000,
    },
    "cues": {
        "vegetation_exg_threshold": 8.0,
        "vegetation_max_lightness": 175.0,
        "shadow_lightness_percentile": 14.0,
        "soil_min_chroma": 9.0,
        "soil_min_a": 132.0,
        "texture_window_px": 11,
    },
    "shadow_direction": {
        # Estimated from the image unless fixed here (degrees, image frame,
        # direction from a building toward its own shadow).
        "override_deg": None,
        "search_step_deg": 5,
        "search_distances_px": [8, 14, 20, 28],
    },
    "segmentation": {
        # Watershed seeds are low-gradient plateaus of the smoothed image.
        "gradient_seed_percentile": 18.0,
        "seed_open_kernel_px": 7,
        "gradient_blur_sigma": 1.2,
    },
    "detector": {
        "name": "classical",
        "params": {
            "min_region_area_px": 400,
            "min_thickness_px": 8.0,
            "elongation_free_ratio": 3.5,
            "elongation_max_ratio": 11.0,
            "accept_score": 0.44,
            "weights": {
                "brightness": 0.28,
                "rectangularity": 0.20,
                "solidity": 0.16,
                "shadow_adjacency": 0.22,
                "texture": 0.06,
                "thickness": 0.12,
                "vegetation_penalty": 0.85,
                "shadow_penalty": 0.90,
                "soil_penalty": 0.55,
                "elongation_penalty": 0.45,
            },
            # Adjacent accepted regions are merged into one building when their
            # mean lightness agrees within this many LAB L units.
            "merge_lightness_tolerance": 26.0,
            "merge_dilate_px": 3,
        },
    },
    "footprint": {
        "min_area_px": 900,
        # Douglas-Peucker tolerance as a fraction of contour perimeter.
        "simplify_fraction": 0.012,
        "min_vertices": 4,
        "max_vertices": 40,
        # Snap edges toward the dominant building axis when close to it.
        "regularize": True,
        "regularize_angle_tolerance_deg": 12.0,
        "fill_hole_area_px": 250,
    },
    "logging": {"level": "INFO"},
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Immutable-ish view over merged configuration with dotted lookup."""

    def __init__(self, data: Mapping[str, Any], source: Path | None = None) -> None:
        self._data = copy.deepcopy(dict(data))
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            log.info("no config file given; using built-in defaults")
            return cls(DEFAULTS, None)
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"config file must contain a YAML mapping: {p}")
        log.info("loaded config from %s", p)
        return cls(_deep_merge(DEFAULTS, raw), p)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted, {})
        return copy.deepcopy(value) if isinstance(value, Mapping) else {}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def fingerprint(self) -> str:
        """Stable hash of the effective configuration, for reproducibility."""
        import hashlib

        blob = json.dumps(self._data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
