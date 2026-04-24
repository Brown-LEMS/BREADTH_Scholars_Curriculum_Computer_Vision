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
from multicam_fall.models import Reconstruction3DFrame  # noqa: E402
from multicam_fall.postprocess import refine_reconstructions  # noqa: E402


def _frame_count(path: Path) -> int:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return len(payload.get("frames", []))
    except Exception:
        return -1


def _choose_reconstruction_file(output_dir: Path, output_json_name: str) -> Path:
    candidates = [
        output_dir / output_json_name,
        output_dir / "reconstruction_refined_debug.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return output_dir / output_json_name
    return max(
        existing,
        key=lambda path: (_frame_count(path), path.stat().st_mtime),
    )


def _prepare_refined_reconstruction(source_path: Path, interpolation_gap: int) -> Path:
    import json

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    frames = [
        Reconstruction3DFrame(
            frame_index=int(frame.get("frame_index", index)),
            timestamp_sec=float(frame.get("timestamp_sec", 0.0)),
            joints=frame.get("joints", {}),
            supporting_cameras=frame.get("supporting_cameras", {}),
        )
        for index, frame in enumerate(payload.get("frames", []))
    ]
    refined_frames = refine_reconstructions(frames, max_interpolation_gap=interpolation_gap)
    payload["frames"] = [
        {
            "frame_index": frame.frame_index,
            "timestamp_sec": frame.timestamp_sec,
            "joints": frame.joints,
            "supporting_cameras": frame.supporting_cameras,
        }
        for frame in refined_frames
    ]
    output_path = source_path.with_name(f"{source_path.stem}_refined_render.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an overview video for a chute sequence.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "project_config_pycharm.json"),
        help="Path to the project config JSON.",
    )
    parser.add_argument(
        "--chute",
        required=True,
        help="Chute sequence name, for example chute01.",
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
    output_dir = Path(config.outputs_root) / args.chute
    reconstruction_path = _choose_reconstruction_file(output_dir, config.output_json_name)
    reconstruction_path = _prepare_refined_reconstruction(
        reconstruction_path,
        interpolation_gap=config.interpolation_gap_frames,
    )

    output_path = output_dir / f"{args.chute}_multicam_overview.mp4"
    saved = render_multicam_overview(
        config=config,
        chute_name=args.chute,
        reconstruction_path=reconstruction_path,
        output_path=output_path,
        panel_size=(args.width, args.height),
        frame_stride=args.frame_stride,
    )
    print(f"Saved overview video to {saved}")


if __name__ == "__main__":
    run()
