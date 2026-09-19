"""Command line interface.

    satrecon analyze  input/site.jpg --config config/site.yaml
    satrecon review   data/site.json
    satrecon generate data/site.json     # 3D geometry -> GLB / OBJ
    satrecon detectors
    satrecon verify   data/site.json      # reproducibility check
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

    generate = subparsers.add_parser(
        "generate", help="build 3D geometry from a reviewed scene and export it"
    )
    generate.add_argument("scene", help="scene JSON produced by `analyze`")
    generate.add_argument("--out-dir", "-o", help="output directory (default output/<name>/model)")
    generate.add_argument(
        "--formats", default="glb,obj",
        help="comma-separated: glb, obj, gltf (default glb,obj)",
    )
    generate.add_argument(
        "--default-height", type=float, default=3.0,
        help="placeholder height (metres if scaled, else pixels) for footprints "
             "with no measured height (default 3.0)",
    )
    _add_common(generate)

    verify = subparsers.add_parser("verify", help="re-run analysis and compare the result hash")
    verify.add_argument("scene", help="scene JSON to reproduce")
    _add_common(verify)

    subparsers.add_parser("detectors", help="list registered detectors")
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
    if summary["heights_estimated"]:
        print(f"heights     {summary['with_height']}/{summary['buildings']} from shadows "
              f"(sun {summary['sun_source']})")
    else:
        print("heights     not estimated (supply scale + sun elevation to enable)")
    print(f"result hash {summary['result_hash']}")
    print()
    print("Review the footprints and heights, then: satrecon generate " + str(scene_path))
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


def cmd_generate(args) -> int:
    from .geometry import export as geo_export
    from .geometry import mesh as geo_mesh
    from .model import Scene

    _configure(args)
    scene = Scene.load(args.scene)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - {"glb", "obj", "gltf"}
    if unknown:
        log.error("unknown export format(s): %s", ", ".join(sorted(unknown)))
        return 2

    model_mesh = geo_mesh.scene_to_mesh(
        scene.buildings, scene.scale, default_height_units=args.default_height
    )
    if model_mesh.is_empty:
        log.error("no geometry to export - the scene has no usable footprints")
        return 2

    units = "meters" if scene.scale.known else "pixels"
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_DIR / scene.meta.name / "model"
    writers = {"glb": geo_export.write_glb, "obj": geo_export.write_obj, "gltf": geo_export.write_gltf}
    written: list[Path] = []
    for fmt in formats:
        written.append(writers[fmt](model_mesh, out_dir / f"{scene.meta.name}.{fmt}", units=units))

    measured = sum(1 for b in scene.buildings if b.estimated_height_m is not None)
    print()
    for path in written:
        print(f"export      {path}")
    print(f"buildings   {len(scene.buildings)}  ({measured} with measured heights, "
          f"{len(scene.buildings) - measured} at placeholder height)")
    print(f"units       {units}"
          + ("" if scene.scale.known else "  (no scale supplied - geometry is in pixels)"))
    print(f"triangles   {len(model_mesh.faces)}")
    if not scene.sun.elevation_deg:
        print()
        print("No sun elevation was available, so heights were not estimated and every")
        print("building uses the placeholder height. Supply geo.center_lat/lon + "
              "sun.timestamp, or sun.elevation_deg, then re-run analyze.")
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
    configure("INFO")
    from .detect import available_detectors
    from .detect import classical  # noqa: F401  (registration side effect)

    for name in available_detectors():
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "analyze": cmd_analyze,
        "review": cmd_review,
        "generate": cmd_generate,
        "verify": cmd_verify,
        "detectors": cmd_detectors,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
