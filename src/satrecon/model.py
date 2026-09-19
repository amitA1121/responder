"""Persistent data model.

Source data (what was observed) is kept separate from generated geometry
(the 3D mesh, built on demand from this scene by geometry/).

Every inferred number carries a `source` and a `confidence` so that nothing in
the file can be mistaken for a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

from . import MODEL_VERSION, __version__

# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------

Point = tuple[float, float]


@dataclass
class ScaleInfo:
    """Pixel-to-metre conversion and where it came from.

    `meters_per_pixel is None` means the scale is genuinely unknown; consumers
    must then present dimensions in pixels, or as relative estimates only.
    """

    meters_per_pixel: float | None = None
    source: str = "unknown"          # user_explicit | user_reference | gsd | geo | unknown
    confidence: float = 0.0
    note: str = "Scale unknown - dimensions are relative estimates."

    @property
    def known(self) -> bool:
        return self.meters_per_pixel is not None and self.meters_per_pixel > 0

    def px_to_m(self, pixels: float) -> float | None:
        return pixels * self.meters_per_pixel if self.known else None


@dataclass
class Confidence:
    """Per-aspect confidence in [0, 1].

    `None` means "not estimated in this run" and is excluded from `overall()`.
    That is deliberately different from 0.0, which means "estimated, and we
    have no faith in it".  Conflating the two would make an unknown scale look
    like a bad footprint - so an aspect that was never attempted is omitted
    from the average and reported as unknown in its own right.
    """

    footprint: float = 0.0
    scale: float | None = None
    height: float | None = None
    floors: float | None = None

    def overall(self) -> float:
        parts = [v for v in (self.footprint, self.scale, self.height, self.floors) if v is not None]
        return round(sum(parts) / len(parts), 4) if parts else 0.0

    def band(self) -> str:
        value = self.overall()
        if value >= 0.66:
            return "HIGH"
        if value >= 0.40:
            return "MEDIUM"
        return "LOW"


@dataclass
class BuildingDetection:
    """Raw output of a BuildingDetector, in source image pixel coordinates."""

    id: str
    contour: list[Point]                  # dense contour, as detected
    bounding_box: tuple[int, int, int, int]   # x, y, w, h
    confidence: float
    orientation_deg: float                # long-axis angle, image frame, CCW from +x
    area_px: float
    detector: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Building:
    """A building candidate after footprint cleanup.

    Geometry and footprint confidence come from detection; height and floor
    fields are filled by the shadow-based height stage when a scale and a sun
    elevation are available, and stay null otherwise.  Roof type is not yet
    inferred (assumed flat at extrusion time).
    """

    id: str
    footprint: list[Point]                # simplified polygon, image pixels
    contour: list[Point] = field(default_factory=list)   # original, unsimplified
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    orientation_deg: float = 0.0
    area_px: float = 0.0
    length_px: float = 0.0
    width_px: float = 0.0
    length_m: float | None = None
    width_m: float | None = None
    area_m2: float | None = None
    estimated_height_m: float | None = None
    height_source: str = "not_estimated"
    estimated_floors: int | None = None
    floors_source: str = "not_estimated"
    roof_type: str = "unknown"
    roof_source: str = "not_inferred"
    confidence: Confidence = field(default_factory=Confidence)
    evidence: dict[str, Any] = field(default_factory=dict)
    # True once a human has edited this building; the pipeline must not
    # silently overwrite such records on a later run.
    user_edited: bool = False

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"]["overall"] = self.confidence.overall()
        data["confidence"]["band"] = self.confidence.band()
        return data


@dataclass
class SceneMeta:
    """Everything needed to reproduce and compare a run."""

    name: str
    image_path: str
    image_sha256: str
    image_width: int
    image_height: int
    config_fingerprint: str
    detector: str
    model_version: str = MODEL_VERSION
    tool_version: str = __version__
    generated_at: str = ""
    # Hash of the result excluding the timestamp: identical input plus
    # identical config must reproduce an identical result_hash.
    result_hash: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class SunInfo:
    """Solar position used for height estimation and where it came from."""

    elevation_deg: float | None = None
    azimuth_deg: float | None = None
    source: str = "unknown"          # user_explicit | computed | unknown
    confidence: float = 0.0
    # Shadow direction the sun implies, image frame, for cross-checking the
    # direction recovered from the image. None when north or azimuth is unknown.
    expected_shadow_direction_deg: float | None = None


@dataclass
class Scene:
    meta: SceneMeta
    scale: ScaleInfo
    buildings: list[Building] = field(default_factory=list)
    shadow_direction_deg: float | None = None
    north_offset_deg: float | None = None
    sun: SunInfo = field(default_factory=SunInfo)
    config: dict[str, Any] = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "scene": {
                "name": self.meta.name,
                "image": {
                    "path": self.meta.image_path,
                    "sha256": self.meta.image_sha256,
                    "width": self.meta.image_width,
                    "height": self.meta.image_height,
                },
                "scale": asdict(self.scale),
                "shadowDirectionDeg": self.shadow_direction_deg,
                "northOffsetDeg": self.north_offset_deg,
                "sun": asdict(self.sun),
                "notes": self.meta.notes,
            },
            "provenance": {
                "toolVersion": self.meta.tool_version,
                "modelVersion": self.meta.model_version,
                "detector": self.meta.detector,
                "configFingerprint": self.meta.config_fingerprint,
                "generatedAt": self.meta.generated_at,
                "resultHash": self.meta.result_hash,
            },
            "config": self.config,
            "buildings": [b.to_json() for b in self.buildings],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2))
        return p

    # -- deserialisation --------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> "Scene":
        data = json.loads(Path(path).read_text())
        scene_node = data["scene"]
        prov = data.get("provenance", {})
        image = scene_node["image"]
        meta = SceneMeta(
            name=scene_node["name"],
            image_path=image["path"],
            image_sha256=image.get("sha256", ""),
            image_width=image["width"],
            image_height=image["height"],
            config_fingerprint=prov.get("configFingerprint", ""),
            detector=prov.get("detector", "unknown"),
            model_version=prov.get("modelVersion", ""),
            tool_version=prov.get("toolVersion", ""),
            generated_at=prov.get("generatedAt", ""),
            result_hash=prov.get("resultHash", ""),
            notes=scene_node.get("notes", []),
        )
        scale_raw = dict(scene_node.get("scale", {}))
        scale = ScaleInfo(**{k: scale_raw[k] for k in scale_raw if k in ScaleInfo.__dataclass_fields__})
        buildings: list[Building] = []
        for raw in data.get("buildings", []):
            conf_raw = dict(raw.get("confidence", {}))
            conf = Confidence(
                footprint=conf_raw.get("footprint", 0.0),
                scale=conf_raw.get("scale", 0.0),
                height=conf_raw.get("height"),
                floors=conf_raw.get("floors"),
            )
            fields = {k: v for k, v in raw.items() if k in Building.__dataclass_fields__}
            fields["confidence"] = conf
            fields["footprint"] = [tuple(p) for p in raw.get("footprint", [])]
            fields["contour"] = [tuple(p) for p in raw.get("contour", [])]
            bbox = raw.get("bounding_box") or (0, 0, 0, 0)
            fields["bounding_box"] = tuple(bbox)
            buildings.append(Building(**fields))
        sun_raw = dict(scene_node.get("sun", {}))
        sun = SunInfo(**{k: sun_raw[k] for k in sun_raw if k in SunInfo.__dataclass_fields__})
        return Scene(
            meta=meta,
            scale=scale,
            buildings=buildings,
            shadow_direction_deg=scene_node.get("shadowDirectionDeg"),
            north_offset_deg=scene_node.get("northOffsetDeg"),
            sun=sun,
            config=data.get("config", {}),
        )


def result_hash(buildings: Sequence[Building], scale: ScaleInfo) -> str:
    """Deterministic digest of the pipeline result, excluding wall-clock time."""
    import hashlib

    payload = {
        "scale": asdict(scale),
        "buildings": [
            {
                "id": b.id,
                "footprint": [[round(x, 3), round(y, 3)] for x, y in b.footprint],
                "orientation": round(b.orientation_deg, 4),
                "area_px": round(b.area_px, 3),
                # Phase 2: identical input and config must reproduce identical
                # heights, so the estimate is part of the reproducibility hash.
                "height_m": None if b.estimated_height_m is None else round(b.estimated_height_m, 3),
            }
            for b in buildings
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
