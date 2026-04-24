from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ProjectConfig:
    dataset_root: str
    outputs_root: str
    default_chute: str
    cameras: List[str] = field(default_factory=lambda: [f"cam{i}" for i in range(1, 9)])
    calibration_file: Optional[str] = None
    pose_backend: str = "mediapipe"
    pose_confidence: float = 0.5
    sample_every_n_frames: int = 1
    max_frames: Optional[int] = 300
    enable_feature_fundamental_matrix: bool = True
    use_calibration_fundamental_matrix: bool = True
    epipolar_threshold_px: float = 4.0
    reprojection_threshold_px: float = 20.0
    min_views_triangulation: int = 2
    min_triangulation_angle_deg: float = 2.5
    sync_reference_camera: str = "cam1"
    sync_signal_joint: str = "mid_shoulder"
    sync_search_radius: int = 20
    prefer_sequence_delays: bool = True
    enable_postprocess: bool = True
    interpolation_gap_frames: int = 3
    output_json_name: str = "reconstruction.json"


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.resolve().parent

    for key in ("dataset_root", "outputs_root", "calibration_file"):
        value = data.get(key)
        if not value:
            continue
        expanded = Path(os.path.expanduser(value))
        if not expanded.is_absolute():
            expanded = (base_dir / expanded).resolve()
        data[key] = str(expanded)

    return ProjectConfig(**data)
