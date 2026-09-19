"""Manual detector for imported building footprints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..model import BuildingDetection
from .base import BuildingDetector, DetectionContext, DetectorResult
from .registry import register_detector


def _as_float_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"expected a point pair, got: {value!r}")
    return float(value[0]), float(value[1])


def _iter_feature_polygons(payload: Any) -> Iterable[list[tuple[float, float]]]:
    if payload is None:
        return

    if isinstance(payload, list):
        for item in payload:
            yield from _iter_feature_polygons(item)
        return

    if not isinstance(payload, dict):
        return

    kind = payload.get("type")
    if kind == "FeatureCollection":
        for feature in payload.get("features", []):
            yield from _iter_feature_polygons(feature)
        return

    if kind == "Feature":
        geometry = payload.get("geometry")
        if geometry is not None:
            yield from _iter_feature_polygons(geometry)
        return

    if kind == "GeometryCollection":
        for geom in payload.get("geometries", []):
            yield from _iter_feature_polygons(geom)
        return

    if kind == "MultiPolygon":
        for polygon in payload.get("coordinates", []):
            for ring in polygon:
                yield [tuple(point[:2]) for point in ring]
        return

    if kind == "Polygon":
        for ring in payload.get("coordinates", []):
            yield [tuple(point[:2]) for point in ring]
        return

    if "coordinates" in payload:
        coords = payload["coordinates"]
        if isinstance(coords, list) and coords and isinstance(coords[0], (list, tuple)):
            if coords[0] and isinstance(coords[0][0], (int, float)):
                yield [tuple(point) for point in coords]
            elif len(coords) > 0 and isinstance(coords[0], list) and coords[0] and isinstance(coords[0][0], (list, tuple)):
                for ring in coords:
                    if ring:
                        yield [tuple(point[:2]) for point in ring]
        return

    if "features" in payload:
        for feature in payload["features"]:
            yield from _iter_feature_polygons(feature)


def _normalize_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = [_as_float_pair(pt) for pt in ring]
    if len(points) < 3:
        return []
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def _convert_to_working_coords(points: list[tuple[float, float]], params: dict[str, Any], context: DetectionContext) -> list[tuple[float, float]]:
    if not params.get("source_coords") and params.get("coord_space") not in {"source", "source_pixels"}:
        return list(points)

    scale_factor = params.get("scale_factor")
    if scale_factor is None:
        scale_factor = params.get("image_scale_factor")
    if scale_factor is None:
        scale_factor = getattr(context, "scale_factor", 1.0)
    if scale_factor in (None, 0):
        return list(points)
    try:
        factor = float(scale_factor)
    except (TypeError, ValueError):
        return list(points)
    if factor <= 0:
        return list(points)
    return [(x / factor, y / factor) for x, y in points]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return abs(sum(xs[i] * ys[(i + 1) % len(xs)] - ys[i] * xs[(i + 1) % len(xs)] for i in range(len(xs)))) / 2.0


class ManualDetector(BuildingDetector):
    name = "manual"

    def detect(self, context: DetectionContext) -> DetectorResult:
        path_value = self.params.get("path")
        if not path_value:
            raise ValueError("manual detector requires detector.params.path to a GeoJSON/JSON footprint file")

        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"manual detector footprint file not found: {path}")

        payload = json.loads(path.read_text())
        detections: list[BuildingDetection] = []
        for index, polygon in enumerate(_iter_feature_polygons(payload), start=1):
            ring = _normalize_ring(list(polygon))
            if len(ring) < 4:
                continue
            points = _convert_to_working_coords(ring, self.params, context)
            if len(points) < 4:
                continue
            contour_np = np.asarray(points, dtype=np.float32)
            area = float(_polygon_area(points))
            if area <= 0.0:
                continue

            rect = cv2.minAreaRect(contour_np)
            (_, _), (rw, rh), angle = rect
            orientation = (-angle if rw >= rh else -(angle + 90.0)) % 180.0
            x, y, w, h = cv2.boundingRect(contour_np)
            detections.append(
                BuildingDetection(
                    id=f"D{index:03d}",
                    contour=[(float(px), float(py)) for px, py in points],
                    bounding_box=(int(x), int(y), int(w), int(h)),
                    confidence=float(self.params.get("confidence", 1.0)),
                    orientation_deg=float(orientation),
                    area_px=area,
                    detector=self.name,
                    evidence={
                        "source": "geojson",
                        "path": str(path),
                        "coord_space": self.params.get("coord_space", "working"),
                    },
                )
            )

        if not detections:
            return DetectorResult(
                detections=[],
                debug_layers={},
                notes=[f"manual detector loaded {path} but no usable polygons were found"],
            )

        return DetectorResult(
            detections=detections,
            debug_layers={},
            notes=[f"manual detector imported {len(detections)} polygon(s) from {path}"],
        )


register_detector("manual", ManualDetector)
