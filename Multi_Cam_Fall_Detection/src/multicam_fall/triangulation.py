from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .calibration import camera_center, camera_space_depth, projection_matrix
from .epipolar import select_consistent_cameras
from .models import CameraCalibration, Reconstruction3DFrame


def triangulate_point_dlt(
    observations: Iterable[Tuple[np.ndarray, np.ndarray, float]]
) -> np.ndarray | None:
    rows = []
    for point_xy, proj, weight in observations:
        x, y = point_xy
        rows.append(weight * (x * proj[2] - proj[0]))
        rows.append(weight * (y * proj[2] - proj[1]))
    if len(rows) < 4:
        return None
    design = np.asarray(rows, dtype=float)
    _, _, vh = np.linalg.svd(design)
    homogeneous = vh[-1]
    if abs(homogeneous[-1]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[-1]


def project_point(projection: np.ndarray, point_3d: np.ndarray) -> np.ndarray | None:
    homogeneous = np.append(point_3d, 1.0)
    projected = projection @ homogeneous
    if abs(projected[2]) < 1e-12:
        return None
    return projected[:2] / projected[2]


def reprojection_error(
    point_3d: np.ndarray,
    image_point: np.ndarray,
    projection: np.ndarray,
) -> float:
    projected = project_point(projection, point_3d)
    if projected is None:
        return float("inf")
    return float(np.linalg.norm(projected - image_point))


def robust_triangulate_point(
    observations: Dict[str, np.ndarray],
    projection_mats: Dict[str, np.ndarray],
    scores: Dict[str, float],
    calibrations: Dict[str, CameraCalibration],
    min_views: int,
    reprojection_threshold_px: float,
    min_triangulation_angle_deg: float,
) -> Tuple[np.ndarray | None, List[str]]:
    camera_ids = list(observations)
    if len(camera_ids) < min_views:
        return None, []

    candidates: List[Tuple[np.ndarray, List[str]]] = []
    for cam_a, cam_b in combinations(camera_ids, 2):
        candidate = triangulate_point_dlt(
            [
                (observations[cam_a], projection_mats[cam_a], max(scores[cam_a], 1e-3)),
                (observations[cam_b], projection_mats[cam_b], max(scores[cam_b], 1e-3)),
            ]
        )
        if candidate is not None:
            candidates.append((candidate, [cam_a, cam_b]))

    all_views_candidate = triangulate_point_dlt(
        [
            (observations[camera_id], projection_mats[camera_id], max(scores[camera_id], 1e-3))
            for camera_id in camera_ids
        ]
    )
    if all_views_candidate is not None:
        candidates.append((all_views_candidate, camera_ids))

    best_point = None
    best_inliers: List[str] = []
    best_error = float("inf")

    for candidate, _ in candidates:
        inliers = []
        errors = []
        for camera_id in camera_ids:
            depth = camera_space_depth(candidate, calibrations[camera_id])
            if depth is not None and depth <= 0:
                continue
            error = reprojection_error(candidate, observations[camera_id], projection_mats[camera_id])
            if error <= reprojection_threshold_px:
                inliers.append(camera_id)
                errors.append(error)
        if len(inliers) < min_views:
            continue
        if max_triangulation_angle_deg(candidate, inliers, calibrations) < min_triangulation_angle_deg:
            continue
        mean_error = float(np.mean(errors)) if errors else float("inf")
        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and mean_error < best_error
        ):
            best_point = candidate
            best_inliers = inliers
            best_error = mean_error

    if best_point is None or len(best_inliers) < min_views:
        return None, []

    refined_point = triangulate_point_dlt(
        [
            (observations[camera_id], projection_mats[camera_id], max(scores[camera_id], 1e-3))
            for camera_id in best_inliers
        ]
    )
    if refined_point is None:
        return None, []
    if max_triangulation_angle_deg(refined_point, best_inliers, calibrations) < min_triangulation_angle_deg:
        return None, []
    return refined_point, best_inliers


def max_triangulation_angle_deg(
    point_3d: np.ndarray,
    camera_ids: Iterable[str],
    calibrations: Dict[str, CameraCalibration],
) -> float:
    centers = []
    for camera_id in camera_ids:
        center = camera_center(calibrations[camera_id])
        if center is not None:
            centers.append(center)
    if len(centers) < 2:
        return 0.0

    point = np.asarray(point_3d, dtype=float)
    max_angle = 0.0
    for center_a, center_b in combinations(centers, 2):
        ray_a = point - center_a
        ray_b = point - center_b
        norm_a = np.linalg.norm(ray_a)
        norm_b = np.linalg.norm(ray_b)
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            continue
        cosine = float(np.dot(ray_a, ray_b) / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle > max_angle:
            max_angle = angle
    return max_angle


def reconstruct_frame(
    frame_index: int,
    timestamp_sec: float,
    per_camera_landmarks: Dict[str, Dict[str, Tuple[np.ndarray, float]]],
    calibrations: Dict[str, CameraCalibration],
    pairwise_fundamentals: Dict[Tuple[str, str], np.ndarray],
    min_views: int,
    epipolar_threshold_px: float,
    reprojection_threshold_px: float,
    min_triangulation_angle_deg: float,
) -> Reconstruction3DFrame:
    output = Reconstruction3DFrame(frame_index=frame_index, timestamp_sec=timestamp_sec)
    all_joint_names = sorted(
        {
            joint_name
            for camera_landmarks in per_camera_landmarks.values()
            for joint_name in camera_landmarks
        }
    )

    for joint_name in all_joint_names:
        observations: Dict[str, np.ndarray] = {}
        scores: Dict[str, float] = {}
        projection_mats: Dict[str, np.ndarray] = {}
        for camera_id, landmarks in per_camera_landmarks.items():
            item = landmarks.get(joint_name)
            calibration = calibrations.get(camera_id)
            if item is None or calibration is None:
                continue
            proj = projection_matrix(calibration)
            if proj is None:
                continue
            observations[camera_id] = item[0]
            scores[camera_id] = item[1]
            projection_mats[camera_id] = proj

        consistent_cameras = select_consistent_cameras(
            joint_name=joint_name,
            observations=observations,
            pairwise_fundamentals=pairwise_fundamentals,
            threshold_px=epipolar_threshold_px,
        )

        if len(consistent_cameras) < min_views:
            continue

        point_3d, inlier_cameras = robust_triangulate_point(
            observations={camera_id: observations[camera_id] for camera_id in consistent_cameras},
            projection_mats={camera_id: projection_mats[camera_id] for camera_id in consistent_cameras},
            scores={camera_id: scores[camera_id] for camera_id in consistent_cameras},
            calibrations={camera_id: calibrations[camera_id] for camera_id in consistent_cameras},
            min_views=min_views,
            reprojection_threshold_px=reprojection_threshold_px,
            min_triangulation_angle_deg=min_triangulation_angle_deg,
        )
        if point_3d is None or len(inlier_cameras) < min_views:
            continue
        output.joints[joint_name] = point_3d.astype(float).tolist()
        output.supporting_cameras[joint_name] = inlier_cameras

    return output
