"""glTF 2.0 / GLB writer.

Written directly against the spec rather than through a library: GLB is a JSON
chunk plus a binary chunk, and doing it here keeps the dependency list at four
packages and the output entirely under our control.

Conventions:
  * Units are metres. Unity and Godot import 1:1; Unreal wants a 100x scale.
  * Scene space is Z-up; glTF is Y-up, so (x, y, z) -> (x, z, -y), which
    preserves right-handedness and keeps roof normals pointing up.
  * Each building is its own node carrying its metadata in `extras`, so a
    building stays selectable and identifiable after import.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from ..mesh import build_building, build_ground, build_roads
from ..model import Scene
from .base import ThreeDExporter, register_exporter

log = get_logger(__name__)

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

DEFAULT_COLORS = {
    "ground": [0.44, 0.45, 0.40, 1.0],
    "road": [0.27, 0.27, 0.29, 1.0],
    "default": [0.78, 0.76, 0.72, 1.0],
}


def _to_gltf_axes(points: np.ndarray) -> np.ndarray:
    """Z-up (x, y, z) -> Y-up (x, z, -y)."""
    if len(points) == 0:
        return points
    out = np.empty_like(points)
    out[:, 0] = points[:, 0]
    out[:, 1] = points[:, 2]
    out[:, 2] = -points[:, 1]
    return out


class _BufferBuilder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _pad(self) -> None:
        while len(self.blob) % 4:
            self.blob.append(0)

    def add_vec3(self, data: np.ndarray) -> int:
        self._pad()
        payload = np.ascontiguousarray(data, dtype=np.float32)
        offset = len(self.blob)
        self.blob.extend(payload.tobytes())
        self.views.append({
            "buffer": 0, "byteOffset": offset, "byteLength": payload.nbytes, "target": 34962,
        })
        self.accessors.append({
            "bufferView": len(self.views) - 1,
            "componentType": 5126,          # FLOAT
            "count": int(len(payload)),
            "type": "VEC3",
            "min": [float(v) for v in payload.min(axis=0)],
            "max": [float(v) for v in payload.max(axis=0)],
        })
        return len(self.accessors) - 1

    def add_indices(self, data: np.ndarray) -> int:
        self._pad()
        payload = np.ascontiguousarray(data, dtype=np.uint32)
        offset = len(self.blob)
        self.blob.extend(payload.tobytes())
        self.views.append({
            "buffer": 0, "byteOffset": offset, "byteLength": payload.nbytes, "target": 34963,
        })
        self.accessors.append({
            "bufferView": len(self.views) - 1,
            "componentType": 5125,          # UNSIGNED_INT
            "count": int(len(payload)),
            "type": "SCALAR",
            "min": [int(payload.min())] if len(payload) else [0],
            "max": [int(payload.max())] if len(payload) else [0],
        })
        return len(self.accessors) - 1


class GLBExporter(ThreeDExporter):
    name = "glb"
    extension = ".glb"

    def export(self, scene: Scene, out_path: Path) -> Path:
        include_ground = bool(self.params.get("include_ground", True))
        include_roads = bool(self.params.get("include_roads", True))
        roof_pitch = float(self.params.get("roof_pitch_m", 2.6))
        default_height = self.params.get("default_height_m")

        buffers = _BufferBuilder()
        materials: list[dict] = []
        material_index: dict[str, int] = {}
        meshes: list[dict] = []
        nodes: list[dict] = []

        def material_for(key: str, color: list[float]) -> int:
            if key not in material_index:
                materials.append({
                    "name": key,
                    "pbrMetallicRoughness": {
                        "baseColorFactor": color,
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.85,
                    },
                    "doubleSided": False,
                })
                material_index[key] = len(materials) - 1
            return material_index[key]

        def add_mesh(mesh, name: str, material_key: str, color: list[float], extras: dict) -> None:
            if len(mesh.positions) == 0:
                return
            position_accessor = buffers.add_vec3(_to_gltf_axes(mesh.positions))
            normal_accessor = buffers.add_vec3(_to_gltf_axes(mesh.normals))
            index_accessor = buffers.add_indices(mesh.indices)
            meshes.append({
                "name": name,
                "primitives": [{
                    "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                    "indices": index_accessor,
                    "material": material_for(material_key, color),
                    "mode": 4,
                }],
            })
            nodes.append({"name": name, "mesh": len(meshes) - 1, "extras": extras})

        skipped = 0
        for building in scene.buildings:
            height = building.estimated_height_m
            if height is None:
                if default_height is None:
                    skipped += 1
                    continue
                height = float(default_height)

            color = building.evidence.get("color")
            key = building.evidence.get("typology", "default")
            rgba = (list(color) + [1.0]) if color else DEFAULT_COLORS["default"]

            mesh = build_building(
                footprint=building.footprint,
                height_m=float(height),
                roof_type=building.roof_type,
                roof_pitch_m=roof_pitch,
                material=key,
                extras={},
            )
            add_mesh(mesh, building.id, key, rgba, {
                "buildingId": building.id,
                "typology": key,
                "floors": building.estimated_floors,
                "heightM": round(float(height), 3),
                "heightSource": building.height_source,
                "floorsSource": building.floors_source,
                "roofType": mesh.extras.get("roofTypeBuilt", building.roof_type),
                "lengthM": building.length_m,
                "widthM": building.width_m,
                "areaM2": building.area_m2,
                "confidence": building.confidence.overall(),
                "confidenceBand": building.confidence.band(),
                "userEdited": building.user_edited,
            })

        if skipped:
            log.warning(
                "%d building(s) had no height and were skipped; "
                "pass --default-height to extrude them anyway", skipped,
            )

        if include_roads and scene.roads:
            polygons = [road["polygon"] for road in scene.roads]
            add_mesh(build_roads(polygons), "roads", "road", DEFAULT_COLORS["road"],
                     {"kind": "roads", "count": len(polygons)})

        if include_ground:
            width = float(scene.meta.image_width)
            depth = float(scene.meta.image_height)
            if scene.scale.known:
                width *= scene.scale.meters_per_pixel
                depth *= scene.scale.meters_per_pixel
            add_mesh(build_ground(width, depth, -0.01), "ground", "ground",
                     DEFAULT_COLORS["ground"], {"kind": "ground", "widthM": width, "depthM": depth})

        document = {
            "asset": {
                "version": "2.0",
                "generator": f"satrecon {scene.meta.tool_version} ({scene.meta.model_version})",
            },
            "scene": 0,
            "scenes": [{"name": scene.meta.name, "nodes": list(range(len(nodes)))}],
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "bufferViews": buffers.views,
            "accessors": buffers.accessors,
            "buffers": [{"byteLength": len(buffers.blob)}],
            "extras": {
                "sceneName": scene.meta.name,
                "units": "meters",
                "upAxis": "Y",
                "resultHash": scene.meta.result_hash,
                "buildingCount": len(scene.buildings),
                "notes": scene.meta.notes,
            },
        }

        out_path = out_path.with_suffix(self.extension)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(_pack_glb(document, bytes(buffers.blob)))
        log.info(
            "exported %d meshes (%d buildings) to %s",
            len(meshes), len(meshes) - int(include_ground) - int(bool(scene.roads and include_roads)),
            out_path,
        )
        return out_path


def _pack_glb(document: dict, blob: bytes) -> bytes:
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)

    total = 12 + 8 + len(json_bytes) + (8 + len(blob) if blob else 0)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(json_bytes), JSON_CHUNK) + json_bytes
    if blob:
        out += struct.pack("<II", len(blob), BIN_CHUNK) + blob
    return bytes(out)


register_exporter("glb", GLBExporter)
