"""Pipeline orchestration for Phase 1.

    image -> preprocess -> cues -> shadow -> segment -> detect -> footprints -> scene

Each arrow is a module boundary; this file only wires them together and
records provenance.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import MODEL_VERSION, __version__
from .config import Config
from .detect import DetectionContext, get_detector
from .detect import classical as _classical  # noqa: F401  (registers "classical")
from .imaging import LoadedImage, load_image
from .logging_setup import get_logger
from .model import Building, Scene, SceneMeta, result_hash
from .stages import cues as cues_stage
from .stages import footprint as footprint_stage
from .stages import preprocess as preprocess_stage
from .stages import scale as scale_stage
from .stages import segmentation as segmentation_stage
from .stages import shadow as shadow_stage

log = get_logger(__name__)


@dataclass
class AnalysisResult:
    scene: Scene
    image: LoadedImage
    debug_layers: dict[str, np.ndarray]
    context: DetectionContext


def analyze(image_path: str | Path, config: Config, scene_name: str | None = None) -> AnalysisResult:
    image = load_image(image_path, int(config.get("preprocess.max_working_edge_px", 2000)))

    scale = scale_stage.resolve(config.section("scale"))

    log.info("stage 1/6 preprocess")
    pre = preprocess_stage.run(image.working_bgr, config.section("preprocess"))

    log.info("stage 2/6 surface cues")
    cue_maps = cues_stage.run(image.working_bgr, pre.source_lab, config.section("cues"))

    log.info("stage 3/6 shadow direction")
    direction = shadow_stage.estimate(cue_maps, config.section("shadow_direction"))
    adjacency = shadow_stage.adjacency_map(cue_maps, direction)

    log.info("stage 4/6 region segmentation")
    segments = segmentation_stage.run(pre, config.section("segmentation"))

    log.info("stage 5/6 building detection")
    detector_name = config.get("detector.name", "classical")
    detector = get_detector(detector_name, config.section("detector.params"))
    context = DetectionContext(
        bgr=image.working_bgr,
        preprocessed=pre,
        cues=cue_maps,
        shadow_direction=direction,
        shadow_adjacency=adjacency,
        segmentation=segments,
    )
    detection_result = detector.detect(context)

    log.info("stage 6/6 footprint extraction")
    buildings: list[Building] = footprint_stage.build(
        detection_result.detections, scale, config.section("footprint"), image.to_source_coords
    )

    notes = _build_notes(scale, direction, detection_result.notes)
    meta = SceneMeta(
        name=scene_name or image.path.stem,
        image_path=str(image.path),
        image_sha256=image.sha256,
        image_width=image.source_size[0],
        image_height=image.source_size[1],
        config_fingerprint=config.fingerprint(),
        detector=detector_name,
        model_version=MODEL_VERSION,
        tool_version=__version__,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    scene = Scene(
        meta=meta,
        scale=scale,
        buildings=buildings,
        shadow_direction_deg=round(direction.angle_deg, 2),
        north_offset_deg=config.get("image.north_offset_deg"),
        config=config.as_dict(),
    )
    scene.meta.result_hash = result_hash(buildings, scale)

    debug_layers: dict[str, np.ndarray] = {
        "gradient": segments.gradient.astype(np.float32) / 255.0,
        "vegetation": cue_maps.vegetation,
        "shadow": cue_maps.shadow,
        "soil": cue_maps.soil,
        "shadow_adjacency": adjacency,
        "preprocessed": pre.bgr,
        "segments": segments.labels,
    }
    debug_layers.update(detection_result.debug_layers)
    return AnalysisResult(scene=scene, image=image, debug_layers=debug_layers, context=context)


def _build_notes(scale, direction, detector_notes: list[str]) -> list[str]:
    notes: list[str] = [
        "Phase 1 output: footprints only. Height, floor count and roof type are "
        "not estimated yet and are recorded as null.",
        "All values are approximations derived from a single image. They are not "
        "measurements of any real structure.",
    ]
    if not scale.known:
        notes.append(scale.note)
    else:
        notes.append(f"Scale {scale.meters_per_pixel:.5f} m/px from {scale.source}.")
    notes.append(
        f"Shadow direction {direction.angle_deg:.0f} deg (image frame, "
        f"{direction.source}, confidence {direction.confidence:.2f})."
    )
    notes.extend(detector_notes)
    return notes


def summarize(scene: Scene) -> dict[str, Any]:
    bands: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for building in scene.buildings:
        bands[building.confidence.band()] += 1
    return {
        "buildings": len(scene.buildings),
        "confidence_bands": bands,
        "scale_known": scene.scale.known,
        "result_hash": scene.meta.result_hash[:16],
    }
