from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from multicam_fall.config import load_config  # noqa: E402
from multicam_fall.pipeline import MultiCameraPipeline  # noqa: E402


def run() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project_config_pycharm.json")
    pipeline = MultiCameraPipeline(config)
    output_path = pipeline.run("chute01")
    print(f"Saved reconstruction output to {output_path}")


if __name__ == "__main__":
    run()
