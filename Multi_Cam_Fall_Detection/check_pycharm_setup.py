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


def main() -> None:
    import cv2
    import mediapipe
    import numpy

    sample_video = PROJECT_ROOT.parent.parent / "Desktop" / "BREADTH" / "Kimia" / "dataset" / "chute01" / "cam1.avi"

    print(f"Python: {sys.executable}")
    print(f"NumPy: {numpy.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"MediaPipe: {mediapipe.__version__}")
    print(f"MPLCONFIGDIR: {os.environ['MPLCONFIGDIR']}")
    print(f"Sample video exists: {sample_video.exists()}")

    if sample_video.exists():
        capture = cv2.VideoCapture(str(sample_video))
        print(f"Video opened: {capture.isOpened()}")
        print(f"FPS: {capture.get(cv2.CAP_PROP_FPS)}")
        print(f"Width: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"Height: {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        capture.release()


if __name__ == "__main__":
    main()
