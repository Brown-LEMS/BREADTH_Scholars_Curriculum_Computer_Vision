from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Landmark2D:
    name: str
    x: float
    y: float
    score: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=float)


@dataclass
class Pose2DFrame:
    camera_id: str
    frame_index: int
    timestamp_sec: float
    landmarks: Dict[str, Landmark2D] = field(default_factory=dict)


@dataclass
class CameraCalibration:
    camera_id: str
    resolution: Tuple[int, int] = (0, 0)
    camera_matrix: Optional[List[List[float]]] = None
    dist_coeffs: Optional[List[float]] = None
    rotation: Optional[List[List[float]]] = None
    translation: Optional[List[float]] = None
    frame_delay: int = 0

    def has_intrinsics(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def has_extrinsics(self) -> bool:
        return self.rotation is not None and self.translation is not None


@dataclass
class Reconstruction3DFrame:
    frame_index: int
    timestamp_sec: float
    joints: Dict[str, List[float]] = field(default_factory=dict)
    supporting_cameras: Dict[str, List[str]] = field(default_factory=dict)

