from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def list_chutes(dataset_root: str) -> List[Path]:
    root = Path(dataset_root)
    return sorted([path for path in root.iterdir() if path.is_dir()])


def discover_camera_videos(chute_dir: str | Path, camera_ids: List[str]) -> Dict[str, Path]:
    base = Path(chute_dir)
    videos: Dict[str, Path] = {}
    for camera_id in camera_ids:
        path = base / f"{camera_id}.avi"
        if path.exists():
            videos[camera_id] = path
    return videos
