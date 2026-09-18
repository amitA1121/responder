"""Residential neighbourhood generator.

    site -> street network -> blocks -> lots -> buildings -> Scene

Output is the same `Scene` the detection path produces, with one honest
difference: nothing here is an estimate. A generated building's height and
floor count are specifications, not inferences, so their sources read
"generated" and their confidences are 1.0.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from .. import MODEL_VERSION, __version__
from ..logging_setup import get_logger
from ..model import Building, Confidence, Scene, SceneMeta, ScaleInfo, result_hash
from . import lots as lots_mod
from . import roads as roads_mod
from . import typology as typology_mod
from .base import SceneGenerator, register_generator
from .roads import Rect

log = get_logger(__name__)


def _rotate(points: np.ndarray, degrees: float, about: tuple[float, float]) -> np.ndarray:
    if abs(degrees) < 1e-9:
        return points
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    origin = np.array(about, dtype=np.float64)
    return (points - origin) @ rotation.T + origin


class NeighborhoodGenerator(SceneGenerator):
    name = "neighborhood"

    def generate(self, seed: int) -> Scene:
        params = self.params
        rng = np.random.default_rng(seed)

        width = float(params.get("site_width_m", 420.0))
        depth = float(params.get("site_depth_m", 320.0))
        rotation = float(params.get("rotation_deg", 0.0))
        centre = (width / 2.0, depth / 2.0)
        site = Rect(0.0, 0.0, width, depth)

        network = roads_mod.generate(site, params.get("roads", {}), rng)
        log.info("street network: %d blocks, %d roads", len(network.blocks), len(network.roads))

        typologies = dict(typology_mod.DEFAULT_TYPOLOGIES)
        typologies.update(params.get("typologies", {}) or {})
        lot_params = params.get("lots", {})
        density_falloff = float(params.get("density_falloff", 1.0))

        placements: list[typology_mod.Placement] = []
        for block in network.blocks:
            # Denser ground gets larger plots, which is what lets the taller
            # typologies fit at all - otherwise every lot is cottage-sized and
            # the density parameter has nothing to act on.
            block_density = self._density_at(block, centre, width, depth, density_falloff)
            block_lots = dict(lot_params)
            block_lots["target_lot_area_m2"] = float(
                lot_params.get("target_lot_area_m2", 900.0)
            ) * (1.0 + 3.2 * block_density)
            for lot in lots_mod.subdivide(block, block_lots, rng):
                density = self._density_at(lot, centre, width, depth, density_falloff)
                placement = typology_mod.place(lot, typologies, density, rng)
                if placement is not None:
                    placements.append(placement)

        log.info("placed %d buildings", len(placements))

        buildings = self._to_buildings(placements, rotation, centre)
        road_polygons = [
            _rotate(np.array(road.rect.corners()), rotation, centre) for road in network.roads
        ]

        # Rotation can push geometry negative; shift everything so the site
        # starts at the origin. Engines and the review page both prefer that.
        offset = self._origin_offset(buildings, road_polygons)
        extent = self._apply_offset(buildings, road_polygons, offset)

        road_records = [
            {
                "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
                "level": road.level,
                "widthM": round(road.rect.w if road.axis == "v" else road.rect.h, 2),
            }
            for road, polygon in zip(network.roads, road_polygons)
        ]

        # Generated coordinates are metres, so one "pixel" is one metre and
        # every derived dimension in the shared schema comes out correct.
        scale = ScaleInfo(
            meters_per_pixel=1.0,
            source="generated",
            confidence=1.0,
            note="Generated scene: coordinates are metres by construction.",
        )
        meta = SceneMeta(
            name=str(params.get("name", "neighborhood")),
            image_path="",
            image_sha256="",
            image_width=int(np.ceil(extent[0])),
            image_height=int(np.ceil(extent[1])),
            config_fingerprint="",
            detector=f"{self.name}(seed={seed})",
            model_version=MODEL_VERSION,
            tool_version=__version__,
            generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            notes=[
                "Procedurally generated scene - not derived from any image or real location.",
                "Heights and floor counts are specifications, not estimates.",
                f"Seed {seed}: the same seed and config reproduce this scene exactly.",
            ],
        )
        scene = Scene(
            meta=meta,
            scale=scale,
            buildings=buildings,
            shadow_direction_deg=None,
            north_offset_deg=None,
            config={"generator": {"name": self.name, "seed": seed, "params": params}},
            roads=road_records,
        )
        scene.meta.result_hash = result_hash(buildings, scale)
        return scene

    @staticmethod
    def _origin_offset(buildings, road_polygons) -> np.ndarray:
        points = [np.array(b.footprint, dtype=np.float64) for b in buildings]
        points.extend(road_polygons)
        if not points:
            return np.zeros(2)
        stacked = np.vstack(points)
        return stacked.min(axis=0)

    @staticmethod
    def _apply_offset(buildings, road_polygons, offset) -> tuple[float, float]:
        maxima = []
        for building in buildings:
            shifted = np.array(building.footprint, dtype=np.float64) - offset
            building.footprint = [(round(float(x), 3), round(float(y), 3)) for x, y in shifted]
            xs, ys = shifted[:, 0], shifted[:, 1]
            building.bounding_box = (
                int(np.floor(xs.min())), int(np.floor(ys.min())),
                int(np.ceil(xs.max() - xs.min())), int(np.ceil(ys.max() - ys.min())),
            )
            maxima.append(shifted.max(axis=0))
        for index, polygon in enumerate(road_polygons):
            road_polygons[index] = polygon - offset
            maxima.append(road_polygons[index].max(axis=0))
        if not maxima:
            return (0.0, 0.0)
        extent = np.vstack(maxima).max(axis=0)
        return (float(extent[0]), float(extent[1]))

    @staticmethod
    def _density_at(lot: Rect, centre, width: float, depth: float, falloff: float) -> float:
        """1.0 at the site centre, falling toward the edges."""
        cx, cy = lot.x + lot.w / 2.0, lot.y + lot.h / 2.0
        dx = (cx - centre[0]) / (width / 2.0)
        dy = (cy - centre[1]) / (depth / 2.0)
        distance = min(float(np.hypot(dx, dy)), 1.0)
        return float(np.clip(1.0 - falloff * distance, 0.0, 1.0))

    def _to_buildings(self, placements, rotation: float, centre) -> list[Building]:
        buildings: list[Building] = []
        for index, placement in enumerate(placements, start=1):
            corners = np.array(placement.footprint.corners(), dtype=np.float64)
            rotated = _rotate(corners, rotation, centre)
            xs, ys = rotated[:, 0], rotated[:, 1]
            long_side = max(placement.footprint.w, placement.footprint.h)
            short_side = min(placement.footprint.w, placement.footprint.h)
            base_angle = 0.0 if placement.footprint.w >= placement.footprint.h else 90.0

            buildings.append(
                Building(
                    id=f"B{index:04d}",
                    footprint=[(round(float(x), 3), round(float(y), 3)) for x, y in rotated],
                    contour=[],
                    bounding_box=(
                        int(np.floor(xs.min())), int(np.floor(ys.min())),
                        int(np.ceil(xs.max() - xs.min())), int(np.ceil(ys.max() - ys.min())),
                    ),
                    orientation_deg=round((base_angle + rotation) % 180.0, 3),
                    area_px=round(placement.footprint.area, 3),
                    length_px=round(long_side, 3),
                    width_px=round(short_side, 3),
                    length_m=round(long_side, 3),
                    width_m=round(short_side, 3),
                    area_m2=round(placement.footprint.area, 3),
                    estimated_height_m=placement.height_m,
                    height_source="generated",
                    estimated_floors=placement.floors,
                    floors_source="generated",
                    roof_type=placement.roof_type,
                    roof_source="generated",
                    confidence=Confidence(footprint=1.0, scale=1.0, height=1.0, floors=1.0),
                    evidence={
                        "typology": placement.typology,
                        "floorHeightM": placement.floor_height_m,
                        "color": [round(c, 3) for c in placement.color],
                    },
                )
            )
        return buildings


register_generator("neighborhood", NeighborhoodGenerator)
