"""Mesh exporters: Wavefront OBJ and binary glTF (GLB).

Both are dependency-free and self-contained. GLB is written by hand rather than
via a library so Phase 2 keeps Phase 1's "no heavy dependencies" property; the
file it produces opens in standard glTF viewers, Blender and three.js.

Units: the mesh is in metres when a scale was known and in pixels otherwise.
The caller passes `units` so the fact travels with the file (OBJ as a comment,
GLB in the asset `extras`) and a viewer never mislabels pixels as metres.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .mesh import Mesh

log = get_logger(__name__)


def write_obj(mesh: Mesh, path: str | Path, units: str = "meters") -> Path:
    """Write an ASCII OBJ. Groups faces per building via `g <id>`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# satrecon Phase 2 export",
        f"# units: {units} (1 unit = 1 {units[:-1] if units.endswith('s') else units})",
        f"# vertices: {len(mesh.vertices)}  triangles: {len(mesh.faces)}",
    ]
    for vx, vy, vz in mesh.vertices:
        lines.append(f"v {vx:.4f} {vy:.4f} {vz:.4f}")

    current = None
    for tri, building_id in zip(mesh.faces, mesh.face_building or [None] * len(mesh.faces)):
        if building_id != current:
            lines.append(f"g {building_id}")
            current = building_id
        # OBJ vertex indices are 1-based.
        lines.append(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}")

    p.write_text("\n".join(lines) + "\n")
    log.info("wrote OBJ %s (%d triangles)", p, len(mesh.faces))
    return p


def _pad4(data: bytes, fill: bytes) -> bytes:
    remainder = len(data) % 4
    return data if remainder == 0 else data + fill * (4 - remainder)


def build_gltf(mesh: Mesh, units: str = "meters") -> tuple[dict, bytes]:
    """Build the glTF 2.0 JSON document with a base64 data-URI buffer.

    Kept separate from GLB packing so it can be tested without byte-wrangling
    and so a text `.gltf` can be emitted as easily as a binary `.glb`.
    """
    positions = mesh.vertices.astype("<f4")
    indices = mesh.faces.reshape(-1).astype("<u4")
    pos_bytes = positions.tobytes()
    idx_bytes = indices.tobytes()

    # Indices first, then positions, each 4-aligned (both are already multiples
    # of 4 bytes, but keep the discipline explicit).
    idx_padded = _pad4(idx_bytes, b"\x00")
    buffer = idx_padded + pos_bytes

    mins = mesh.vertices.min(axis=0).tolist() if len(mesh.vertices) else [0, 0, 0]
    maxs = mesh.vertices.max(axis=0).tolist() if len(mesh.vertices) else [0, 0, 0]

    document = {
        "asset": {
            "version": "2.0",
            "generator": "satrecon",
            "extras": {"units": units},
        },
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0, "name": "satrecon_buildings"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 1},
                        "indices": 0,
                        "material": 0,
                        "mode": 4,   # triangles
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "building",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.78, 0.78, 0.80, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
            }
        ],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": len(idx_padded), "byteLength": len(pos_bytes), "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5125,   # UNSIGNED_INT
                "count": int(indices.size),
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "componentType": 5126,   # FLOAT
                "count": int(len(mesh.vertices)),
                "type": "VEC3",
                "min": [float(v) for v in mins],
                "max": [float(v) for v in maxs],
            },
        ],
    }
    return document, buffer


def write_glb(mesh: Mesh, path: str | Path, units: str = "meters") -> Path:
    """Write a binary glTF (.glb): 12-byte header + JSON chunk + BIN chunk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    document, buffer = build_gltf(mesh, units)
    # In GLB the single buffer is the BIN chunk, so it carries no uri.
    document["buffers"][0].pop("uri", None)

    json_bytes = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = _pad4(buffer, b"\x00")

    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes   # "JSON"
    bin_chunk = struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes      # "BIN\0"
    total = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)                          # "glTF", v2

    p.write_bytes(header + json_chunk + bin_chunk)
    log.info("wrote GLB %s (%d bytes, %d triangles)", p, total, len(mesh.faces))
    return p


def write_gltf(mesh: Mesh, path: str | Path, units: str = "meters") -> Path:
    """Write a text `.gltf` with the buffer embedded as a base64 data URI."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    document, buffer = build_gltf(mesh, units)
    document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(
        buffer
    ).decode("ascii")
    p.write_text(json.dumps(document, indent=2))
    log.info("wrote glTF %s", p)
    return p
