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

from multicam_fall.visualization import save_reconstruction_gif  # noqa: E402


def main() -> None:
    reconstruction_path = PROJECT_ROOT / "outputs" / "chute01" / "reconstruction.json"
    output_path = PROJECT_ROOT / "outputs" / "chute01" / "reconstruction_3d.gif"
    saved = save_reconstruction_gif(
        reconstruction_path=reconstruction_path,
        output_path=output_path,
        fps=10,
        frame_stride=2,
    )
    print(f"Saved 3D visualization to {saved}")


if __name__ == "__main__":
    main()
