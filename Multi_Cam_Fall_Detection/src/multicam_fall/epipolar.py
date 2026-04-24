from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .models import Pose2DFrame


def sampson_distance(point_a: np.ndarray, point_b: np.ndarray, fundamental: np.ndarray) -> float:
    x1 = np.asarray([point_a[0], point_a[1], 1.0], dtype=float)
    x2 = np.asarray([point_b[0], point_b[1], 1.0], dtype=float)
    fx1 = fundamental @ x1
    ftx2 = fundamental.T @ x2
    numerator = float((x2.T @ fundamental @ x1) ** 2)
    denominator = fx1[0] ** 2 + fx1[1] ** 2 + ftx2[0] ** 2 + ftx2[1] ** 2
    if denominator <= 1e-12:
        return float("inf")
    return numerator / denominator


def validate_corresponding_landmarks(
    pose_a: Pose2DFrame,
    pose_b: Pose2DFrame,
    fundamental: np.ndarray,
    threshold_px: float,
) -> Dict[str, bool]:
    valid: Dict[str, bool] = {}
    shared = set(pose_a.landmarks).intersection(pose_b.landmarks)
    for name in shared:
        landmark_a = pose_a.landmarks[name]
        landmark_b = pose_b.landmarks[name]
        valid[name] = sampson_distance(
            landmark_a.as_array(),
            landmark_b.as_array(),
            fundamental,
        ) <= threshold_px**2
    return valid


def select_consistent_cameras(
    joint_name: str,
    observations: Dict[str, np.ndarray],
    pairwise_fundamentals: Dict[Tuple[str, str], np.ndarray],
    threshold_px: float,
) -> List[str]:
    if len(observations) < 2:
        return list(observations)

    adjacency = {camera_id: set() for camera_id in observations}
    camera_ids = list(observations)
    for idx, cam_a in enumerate(camera_ids):
        for cam_b in camera_ids[idx + 1 :]:
            fundamental = pairwise_fundamentals.get((cam_a, cam_b))
            if fundamental is None:
                continue
            distance = sampson_distance(observations[cam_a], observations[cam_b], fundamental)
            if distance <= threshold_px**2:
                adjacency[cam_a].add(cam_b)
                adjacency[cam_b].add(cam_a)

    consistent = [camera_id for camera_id, neighbors in adjacency.items() if neighbors]
    return consistent if consistent else list(observations)
