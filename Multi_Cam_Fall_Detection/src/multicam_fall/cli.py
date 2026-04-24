from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import MultiCameraPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-camera 3D skeleton pipeline for one chute sequence."
    )
    parser.add_argument(
        "--config",
        default="configs/project_config.json",
        help="Path to the project configuration JSON file.",
    )
    parser.add_argument(
        "--chute",
        default=None,
        help="Optional chute override, for example `chute01`.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = MultiCameraPipeline(config)
    output_path = pipeline.run(chute_name=args.chute)
    print(f"Saved reconstruction output to {output_path}")


if __name__ == "__main__":
    main()
