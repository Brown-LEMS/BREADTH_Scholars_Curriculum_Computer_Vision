from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .models import CameraCalibration

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


def load_calibration_file(path: str | Path | None) -> Dict[str, CameraCalibration]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = {}
    for camera_id, item in data.get("cameras", {}).items():
        cameras[camera_id] = CameraCalibration(
            camera_id=camera_id,
            resolution=tuple(item.get("resolution", [0, 0])),
            camera_matrix=item.get("camera_matrix"),
            dist_coeffs=item.get("dist_coeffs"),
            rotation=item.get("rotation"),
            translation=item.get("translation"),
            frame_delay=item.get("frame_delay", 0),
        )
    return cameras


def load_sequence_delays(path: str | Path | None) -> Dict[str, Dict[str, int]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    delays = {}
    for chute_name, item in data.get("sequence_delays", {}).items():
        delays[chute_name] = {camera_id: int(value) for camera_id, value in item.items()}
    return delays


def projection_matrix(calibration: CameraCalibration) -> Optional[np.ndarray]:
    if not (calibration.has_intrinsics() and calibration.has_extrinsics()):
        return None
    k = np.asarray(calibration.camera_matrix, dtype=float)
    r = np.asarray(calibration.rotation, dtype=float)
    t = np.asarray(calibration.translation, dtype=float).reshape(3, 1)
    return k @ np.hstack([r, t])


def camera_space_depth(point_3d: np.ndarray, calibration: CameraCalibration) -> Optional[float]:
    if not calibration.has_extrinsics():
        return None
    r = np.asarray(calibration.rotation, dtype=float)
    t = np.asarray(calibration.translation, dtype=float).reshape(3, 1)
    x = np.asarray(point_3d, dtype=float).reshape(3, 1)
    camera_point = r @ x + t
    return float(camera_point[2, 0])


def camera_center(calibration: CameraCalibration) -> Optional[np.ndarray]:
    if not calibration.has_extrinsics():
        return None
    r = np.asarray(calibration.rotation, dtype=float)
    t = np.asarray(calibration.translation, dtype=float).reshape(3, 1)
    center = -r.T @ t
    return center.reshape(3)


def undistort_frame(frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    if cv2 is None or not calibration.has_intrinsics():
        return frame
    k = np.asarray(calibration.camera_matrix, dtype=float)
    d = np.asarray(calibration.dist_coeffs, dtype=float)
    return cv2.undistort(frame, k, d)


def undistort_points(points_xy: Iterable[Iterable[float]], calibration: CameraCalibration) -> np.ndarray:
    points = np.asarray(list(points_xy), dtype=float)
    if points.size == 0:
        return points.reshape(0, 2)
    if cv2 is None or not calibration.has_intrinsics():
        return points
    k = np.asarray(calibration.camera_matrix, dtype=float)
    d = np.asarray(calibration.dist_coeffs, dtype=float)
    pts = points.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, k, d, P=k)
    return undistorted.reshape(-1, 2)


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec.tolist()
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def fundamental_from_calibration(
    source: Optional[CameraCalibration],
    target: Optional[CameraCalibration],
) -> Optional[np.ndarray]:
    if source is None or target is None:
        return None
    if not (
        source.has_intrinsics()
        and source.has_extrinsics()
        and target.has_intrinsics()
        and target.has_extrinsics()
    ):
        return None

    k1 = np.asarray(source.camera_matrix, dtype=float)
    k2 = np.asarray(target.camera_matrix, dtype=float)
    r1 = np.asarray(source.rotation, dtype=float)
    r2 = np.asarray(target.rotation, dtype=float)
    t1 = np.asarray(source.translation, dtype=float).reshape(3, 1)
    t2 = np.asarray(target.translation, dtype=float).reshape(3, 1)

    r21 = r2 @ np.linalg.inv(r1)
    t21 = t2 - r21 @ t1
    essential = _skew(t21.reshape(3)) @ r21
    fundamental = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)
    norm = np.linalg.norm(fundamental)
    if norm == 0:
        return None
    return fundamental / norm
