from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from multicam_fall.composite_video import render_multicam_overview  # noqa: E402
from multicam_fall.config import load_config  # noqa: E402
from multicam_fall.pipeline import MultiCameraPipeline  # noqa: E402
from render_overview import _prepare_refined_reconstruction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct one chute sequence and render its stabilized overview video."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "project_config_pycharm.json"),
        help="Path to the project config JSON.",
    )
    parser.add_argument(
        "--chute",
        required=True,
        help="Chute sequence name, for example chute02.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Panel width for each tile.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Panel height for each tile.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Stride used when exporting overview frames.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    config = load_config(args.config)

    pipeline = MultiCameraPipeline(config)
    reconstruction_path = pipeline.run(chute_name=args.chute)
    print(f"Saved reconstruction output to {reconstruction_path}")

    refined_path = _prepare_refined_reconstruction(
        Path(reconstruction_path),
        interpolation_gap=config.interpolation_gap_frames,
    )
    output_path = Path(config.outputs_root) / args.chute / f"{args.chute}_multicam_overview.mp4"
    saved = render_multicam_overview(
        config=config,
        chute_name=args.chute,
        reconstruction_path=refined_path,
        output_path=output_path,
        panel_size=(args.width, args.height),
        frame_stride=args.frame_stride,
    )
    print(f"Saved overview video to {saved}")


if __name__ == "__main__":
    run()
