from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multicam_fall.ground_truth import save_ground_truth_json  # noqa: E402

PDF_PATH = Path("/Users/crescent/Desktop/BREADTH/Kimia/technicalReport-2.pdf")
OUTPUT_PATH = PROJECT_ROOT / "configs" / "ground_truth_breadth_report.json"


def main() -> None:
    output_path = save_ground_truth_json(PDF_PATH, OUTPUT_PATH)
    print(f"Saved ground truth annotations to {output_path}")


if __name__ == "__main__":
    main()
