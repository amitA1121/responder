"""Command line interface.

    satrecon analyze input/site.jpg --config config/site.yaml
    satrecon review  data/site.json
    satrecon detectors
    satrecon verify  data/site.json      # reproducibility check

Phase 2 will add `generate` (3D scene) and `export` (GLB / Blender).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import MODEL_VERSION, PHASE, __version__
from .config import Config
from .logging_setup import configure, get_logger

log = get_logger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("output")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", help="YAML configuration file")
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="logging verbosity",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satrecon",
        description=(
            "Satellite/aerial image to approximate 3D scene pipeline "
            f"(phase {PHASE}: detection and footprints)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"satrecon {__version__} ({MODEL_VERSION})")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="detect buildings and extract footprints")
    analyze.add_argument("image", help="input image (png, jpg, jpeg, webp, tif)")
    analyze.add_argument("--out", "-o", help="scene JSON path (default data/<name>.json)")
    analyze.add_argument("--debug-dir", help="directory for debug rasters (default output/<name>)")
    analyze.add_argument("--name", help="scene name (default: image stem)")
    analyze.add_argument("--no-debug", action="store_true", help="skip debug rasters")
    analyze.add_argument("--no-review", action="store_true", help="skip the HTML review page")
    analyze.add_argument("--embed-image", action="store_true",
                         help="inline the image into the review page instead of copying it")
    _add_common(analyze)

    review = subparsers.add_parser("review", help="(re)build the 2D review page from a scene file")
    review.add_argument("scene", help="scene JSON produced by `analyze`")
    review.add_argument("--out", "-o", help="output HTML path")
    review.add_argument("--image", help="override the image path recorded in the scene")
    review.add_argument("--embed-image", action="store_true")
    _add_common(review)

    generate = subparsers.add_parser("generate", help="procedurally generate a scene (no image)")
    generate.add_argument("--seed", type=int, help="override the configured seed")
    generate.add_argument("--out", "-o", help="scene JSON path (default data/<name>.json)")
    generate.add_argument("--name", help="scene name")
    generate.add_argument("--export", action="store_true", help="also write the 3D export")
    generate.add_argument("--no-review", action="store_true", help="skip the 2D review page")
    _add_common(generate)

    export = subparsers.add_parser("export", help="write 3D geometry from a scene file")
    export.add_argument("scene", help="scene JSON")
    export.add_argument("--out", "-o", help="output path (extension set by the exporter)")
    export.add_argument("--format", "-f", default="glb", help="exporter name (default glb)")
    export.add_argument("--default-height", type=float,
                        help="extrude buildings that have no height, using this value in metres")
    export.add_argument("--no-ground", action="store_true")
    export.add_argument("--no-roads", action="store_true")
    _add_common(export)

    verify = subparsers.add_parser("verify", help="re-run analysis and compare the result hash")
    verify.add_argument("scene", help="scene JSON to reproduce")
    _add_common(verify)

    subparsers.add_parser("detectors", help="list registered detectors")
    subparsers.add_parser("generators", help="list registered generators")
    subparsers.add_parser("exporters", help="list registered exporters")
    return parser


def _configure(args) -> Config:
    config = Config.load(getattr(args, "config", None))
    level = getattr(args, "log_level", None) or config.get("logging.level", "INFO")
    configure(level)
    return config


def cmd_analyze(args) -> int:
    from . import pipeline
    from .debug import render, review

    config = _configure(args)
    result = pipeline.analyze(args.image, config, args.name)
    scene = result.scene

    scene_path = Path(args.out) if args.out else DEFAULT_DATA_DIR / f"{scene.meta.name}.json"
    scene.save(scene_path)
    log.info("wrote scene %s", scene_path)

    debug_dir = Path(args.debug_dir) if args.debug_dir else DEFAULT_OUTPUT_DIR / scene.meta.name
    overlay_path = debug_dir / "overlay.png"
    render.render_overlay(scene, result.image.source_bgr, overlay_path)
    render.render_footprints_only(
        scene, (scene.meta.image_width, scene.meta.image_height), debug_dir / "footprints.png"
    )
    if not args.no_debug:
        written = render.write_layers(result.debug_layers, debug_dir / "stages")
        log.info("wrote %d stage rasters to %s", len(written), debug_dir / "stages")

    review_path = None
    if not args.no_review:
        review_path = review.write(
            scene, debug_dir / "review.html", result.image.path, embed_image=args.embed_image
        )

    summary = pipeline.summarize(scene)
    print()
    print(f"scene       {scene_path}")
    print(f"overlay     {overlay_path}")
    if review_path:
        print(f"review      {review_path}")
    print(f"buildings   {summary['buildings']}  "
          f"(HIGH {summary['confidence_bands']['HIGH']} / "
          f"MEDIUM {summary['confidence_bands']['MEDIUM']} / "
          f"LOW {summary['confidence_bands']['LOW']})")
    print(f"scale       {'known' if summary['scale_known'] else 'UNKNOWN - dimensions are relative estimates'}")
    print(f"result hash {summary['result_hash']}")
    print()
    print("Phase 1 only: heights, floor counts and roof types are not estimated.")
    print("Open the review page and confirm the footprints before generating 3D geometry.")
    return 0


def cmd_generate(args) -> int:
    from .generate import get_generator
    from .generate import neighborhood  # noqa: F401  (registers "neighborhood")

    config = _configure(args)
    params = config.section("generator.params")
    if args.name:
        params["name"] = args.name
    seed = args.seed if args.seed is not None else int(config.get("generator.seed", 1))

    generator = get_generator(config.get("generator.name", "neighborhood"), params)
    scene = generator.generate(seed)
    scene.config["configFingerprint"] = config.fingerprint()

    scene_path = Path(args.out) if args.out else DEFAULT_DATA_DIR / f"{scene.meta.name}.json"
    scene.save(scene_path)
    log.info("wrote scene %s", scene_path)

    out_dir = DEFAULT_OUTPUT_DIR / scene.meta.name
    review_path = None
    if not args.no_review:
        from .debug import review as review_mod
        review_path = review_mod.write(scene, out_dir / "review.html", None)

    export_path = None
    if args.export:
        from .export import get_exporter
        from .export import gltf  # noqa: F401  (registers "glb")

        exporter = get_exporter(config.get("export.name", "glb"), config.section("export.params"))
        export_path = exporter.export(scene, out_dir / scene.meta.name)

    floors = [b.estimated_floors for b in scene.buildings if b.estimated_floors]
    heights = [b.estimated_height_m for b in scene.buildings if b.estimated_height_m]
    print()
    print(f"scene       {scene_path}")
    if review_path:
        print(f"review      {review_path}")
    if export_path:
        print(f"export      {export_path}")
    print(f"buildings   {len(scene.buildings)}   roads {len(scene.roads)}")
    if heights:
        print(f"heights     {min(heights):.1f}-{max(heights):.1f} m   "
              f"floors {min(floors)}-{max(floors)}")
    print(f"site        {scene.meta.image_width} x {scene.meta.image_height} m")
    print(f"seed        {seed}   result hash {scene.meta.result_hash[:16]}")
    print()
    print("Generated scene - not derived from any image or real location.")
    return 0


def cmd_export(args) -> int:
    from .export import get_exporter
    from .export import gltf  # noqa: F401
    from .model import Scene

    config = _configure(args)
    scene = Scene.load(args.scene)
    params = config.section("export.params")
    if args.no_ground:
        params["include_ground"] = False
    if args.no_roads:
        params["include_roads"] = False
    if args.default_height is not None:
        params["default_height_m"] = args.default_height

    exporter = get_exporter(args.format, params)
    out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / scene.meta.name / scene.meta.name
    path = exporter.export(scene, out)
    print(path)
    return 0


def cmd_review(args) -> int:
    from .debug import review
    from .model import Scene

    _configure(args)
    scene_path = Path(args.scene)
    scene = Scene.load(scene_path)
    image_path = Path(args.image) if args.image else Path(scene.meta.image_path)
    if not image_path.is_file():
        log.error("image not found: %s (pass --image to override)", image_path)
        return 2
    out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / scene.meta.name / "review.html"
    path = review.write(scene, out, image_path, embed_image=args.embed_image)
    print(path)
    return 0


def cmd_verify(args) -> int:
    from . import pipeline
    from .config import Config as _Config
    from .model import Scene

    _configure(args)
    scene = Scene.load(args.scene)
    config = _Config(scene.config, None)
    configure(config.get("logging.level", "INFO"))
    image_path = Path(scene.meta.image_path)
    if not image_path.is_file():
        log.error("recorded image not found: %s", image_path)
        return 2

    result = pipeline.analyze(image_path, config, scene.meta.name)
    same_image = result.scene.meta.image_sha256 == scene.meta.image_sha256
    same_result = result.scene.meta.result_hash == scene.meta.result_hash
    print(json.dumps({
        "imageHashMatches": same_image,
        "resultHashMatches": same_result,
        "recorded": scene.meta.result_hash[:16],
        "recomputed": result.scene.meta.result_hash[:16],
        "recordedBuildings": len(scene.buildings),
        "recomputedBuildings": len(result.scene.buildings),
    }, indent=2))
    return 0 if (same_image and same_result) else 1


def cmd_detectors(_args) -> int:
    configure("WARNING")
    from .detect import available_detectors
    from .detect import classical  # noqa: F401  (registration side effect)

    for name in available_detectors():
        print(name)
    return 0


def cmd_generators(_args) -> int:
    configure("WARNING")
    from .generate import available_generators
    from .generate import neighborhood  # noqa: F401

    for name in available_generators():
        print(name)
    return 0


def cmd_exporters(_args) -> int:
    configure("WARNING")
    from .export import available_exporters
    from .export import gltf  # noqa: F401

    for name in available_exporters():
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "analyze": cmd_analyze,
        "review": cmd_review,
        "verify": cmd_verify,
        "detectors": cmd_detectors,
        "generate": cmd_generate,
        "export": cmd_export,
        "generators": cmd_generators,
        "exporters": cmd_exporters,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
