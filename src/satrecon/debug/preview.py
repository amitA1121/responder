"""Software preview renderer.

A quick look at the generated geometry without opening an engine. Deliberately
simple: back-face culling, painter's algorithm, flat shading. Good enough to
catch inverted normals, roofs at the wrong height or a collapsed footprint -
which is all it is for.

Not a substitute for the interactive viewer; it just closes the loop fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..mesh import build_building, build_ground, build_roads
from ..model import Scene

log = get_logger(__name__)


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float = 42.0


def _look_at(camera: Camera) -> np.ndarray:
    forward = camera.target - camera.eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, camera.up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ camera.eye
    return view


def _orbit_camera(bounds: tuple[np.ndarray, np.ndarray], azimuth_deg: float,
                  elevation_deg: float, distance_scale: float) -> Camera:
    """Frame the geometry from its actual bounds rather than assuming where it sits."""
    low, high = bounds
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low)) * distance_scale
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    eye = centre + np.array([
        radius * np.cos(el) * np.cos(az),
        radius * np.sin(el),
        radius * np.cos(el) * np.sin(az),
    ])
    return Camera(eye=eye, target=centre, up=np.array([0.0, 1.0, 0.0]))


def _scene_meshes(scene: Scene, roof_pitch: float, default_height: float | None):
    """Yield (positions Y-up, normals, indices, rgb) for everything drawable."""
    for building in scene.buildings:
        height = building.estimated_height_m
        if height is None:
            if default_height is None:
                continue
            height = default_height
        mesh = build_building(
            building.footprint, float(height), building.roof_type, roof_pitch
        )
        if len(mesh.positions) == 0:
            continue
        color = building.evidence.get("color") or [0.78, 0.76, 0.72]
        yield _z_up_to_y_up(mesh.positions), _z_up_to_y_up(mesh.normals), mesh.indices, np.array(color)

    if scene.roads:
        mesh = build_roads([r["polygon"] for r in scene.roads])
        if len(mesh.positions):
            yield _z_up_to_y_up(mesh.positions), _z_up_to_y_up(mesh.normals), mesh.indices, np.array([0.27, 0.27, 0.29])

    width = float(scene.meta.image_width)
    depth = float(scene.meta.image_height)
    ground = build_ground(width, depth, -0.02)
    yield _z_up_to_y_up(ground.positions), _z_up_to_y_up(ground.normals), ground.indices, np.array([0.44, 0.45, 0.40])


def _z_up_to_y_up(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0], out[:, 1], out[:, 2] = points[:, 0], points[:, 2], -points[:, 1]
    return out


def render(
    scene: Scene,
    out_path: Path,
    size: tuple[int, int] = (1400, 900),
    azimuth_deg: float = 35.0,
    elevation_deg: float = 28.0,
    distance_scale: float = 1.15,
    roof_pitch_m: float = 2.6,
    default_height_m: float | None = None,
) -> Path:
    width_px, height_px = size
    drawable = list(_scene_meshes(scene, roof_pitch_m, default_height_m))
    if not drawable:
        raise RuntimeError("nothing to render: the scene has no drawable geometry")

    stacked = np.vstack([positions for positions, _, _, _ in drawable])
    bounds = (stacked.min(axis=0), stacked.max(axis=0))
    camera = _orbit_camera(bounds, azimuth_deg, elevation_deg, distance_scale)
    view = _look_at(camera)

    focal = (height_px / 2.0) / np.tan(np.deg2rad(camera.fov_deg) / 2.0)
    light = np.array([0.45, 0.82, 0.35])
    light = light / np.linalg.norm(light)

    faces: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    for positions, normals, indices, color in drawable:
        homogeneous = np.hstack([positions, np.ones((len(positions), 1))])
        camera_space = (view @ homogeneous.T).T[:, :3]
        normals_camera = (view[:3, :3] @ normals.T).T

        tri = indices.reshape(-1, 3)
        for a, b, c in tri:
            pts = camera_space[[a, b, c]]
            if np.any(pts[:, 2] > -0.5):        # behind or too close to the camera
                continue
            normal = normals_camera[a]
            if normal[2] <= 0.0:                # back-face
                continue
            shade = 0.32 + 0.68 * float(np.clip(np.dot(normals[a], light), 0.0, 1.0))
            rgb = np.clip(color * shade, 0, 1) * 255

            screen = np.empty((3, 2), np.int32)
            screen[:, 0] = (pts[:, 0] * focal / -pts[:, 2] + width_px / 2.0).astype(np.int32)
            screen[:, 1] = (-pts[:, 1] * focal / -pts[:, 2] + height_px / 2.0).astype(np.int32)
            faces.append((float(pts[:, 2].mean()), screen, tuple(int(v) for v in rgb[::-1])))

    # Painter's algorithm: furthest (most negative z) first.
    faces.sort(key=lambda f: f[0])
    canvas = np.full((height_px, width_px, 3), 28, np.uint8)
    for _, screen, color in faces:
        cv2.fillConvexPoly(canvas, screen, color, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    log.info("rendered preview with %d faces to %s", len(faces), out_path)
    return out_path
