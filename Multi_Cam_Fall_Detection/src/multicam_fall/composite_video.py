from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .calibration import load_calibration_file, projection_matrix, undistort_frame
from .config import ProjectConfig
from .dataset import discover_camera_videos
from .pose2d import create_pose_estimator

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


PANEL_CONNECTIONS = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]
PANEL_TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
POSE_TRACK_ALPHA = 0.38
POSE_TRACK_MAX_MISSES = 4
POSE_TRACK_MAX_JUMP_PX = 34.0
SKELETON_CENTER_ALPHA = 0.22


def _load_reconstruction_payload(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _global_view_radius(frames: Iterable[Dict[str, object]]) -> float:
    radii = []
    for frame in frames:
        joints = frame.get("joints", {})
        if not joints:
            continue
        points = np.asarray(list(joints.values()), dtype=float)
        center = _frame_center(joints)
        distances = np.linalg.norm(points - center, axis=1)
        radii.append(float(np.percentile(distances, 90)))
    if not radii:
        return 1.0
    return max(float(np.percentile(np.asarray(radii, dtype=float), 85)), 1.0)


def _frame_center(joints: Dict[str, List[float]]) -> np.ndarray:
    torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    anchors = [joints[name] for name in torso_names if name in joints]
    if len(anchors) >= 2:
        return np.asarray(anchors, dtype=float).mean(axis=0)
    return np.asarray(list(joints.values()), dtype=float).mean(axis=0)


def _project_point(point: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = point - center
    return np.asarray(
        [
            centered[0] + 0.45 * centered[1],
            -centered[2] + 0.18 * centered[1],
        ],
        dtype=float,
    )


def _render_skeleton_panel(
    frame_payload: Dict[str, object],
    panel_size: Tuple[int, int],
    radius: float,
    depth_reference: float,
    display_center: np.ndarray | None = None,
) -> np.ndarray:
    height, width = panel_size[1], panel_size[0]
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    if cv2 is None:
        return canvas

    joints = frame_payload.get("joints", {})
    cv2.putText(
        canvas,
        "3D Skeleton",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    if not joints:
        cv2.putText(
            canvas,
            "No 3D joints",
            (width // 2 - 110, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return canvas

    center = np.asarray(display_center, dtype=float) if display_center is not None else _frame_center(joints)
    points_2d = {}
    scale = min(width, height) * 0.36 / max(radius, 1e-6)
    origin = np.asarray([width / 2.0, height / 2.0 + 20.0], dtype=float)
    floor_z = min(float(point[2]) for point in joints.values())

    grid_step = 0.5 if radius <= 2.5 else 1.0
    grid_extent = max(np.ceil(radius / grid_step) * grid_step, grid_step)
    grid_values = np.arange(-grid_extent, grid_extent + 0.001, grid_step)
    for offset in grid_values:
        line_color = (48, 48, 48) if abs(offset) > 1e-6 else (78, 78, 78)
        start = center + np.asarray([offset, -grid_extent, floor_z - center[2]], dtype=float)
        end = center + np.asarray([offset, grid_extent, floor_z - center[2]], dtype=float)
        start_px = origin + _project_point(start, center) * np.asarray([scale, scale], dtype=float)
        end_px = origin + _project_point(end, center) * np.asarray([scale, scale], dtype=float)
        cv2.line(canvas, tuple(start_px.astype(int)), tuple(end_px.astype(int)), line_color, 1, cv2.LINE_AA)
        start = center + np.asarray([-grid_extent, offset, floor_z - center[2]], dtype=float)
        end = center + np.asarray([grid_extent, offset, floor_z - center[2]], dtype=float)
        start_px = origin + _project_point(start, center) * np.asarray([scale, scale], dtype=float)
        end_px = origin + _project_point(end, center) * np.asarray([scale, scale], dtype=float)
        cv2.line(canvas, tuple(start_px.astype(int)), tuple(end_px.astype(int)), line_color, 1, cv2.LINE_AA)

    for joint_name, point in joints.items():
        projected = _project_point(np.asarray(point, dtype=float), center)
        pixel = origin + projected * np.asarray([scale, scale], dtype=float)
        points_2d[joint_name] = pixel.astype(int)

    for joint_a, joint_b in PANEL_CONNECTIONS:
        if joint_a not in points_2d or joint_b not in points_2d:
            continue
        cv2.line(
            canvas,
            tuple(points_2d[joint_a]),
            tuple(points_2d[joint_b]),
            (58, 120, 245),
            3,
            cv2.LINE_AA,
        )

    for point in points_2d.values():
        cv2.circle(canvas, tuple(point), 5, (226, 92, 83), -1, cv2.LINE_AA)

    ruler_start = (22, height - 52)
    ruler_end = (int(22 + scale), height - 52)
    cv2.line(canvas, ruler_start, ruler_end, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(canvas, (ruler_start[0], ruler_start[1] - 6), (ruler_start[0], ruler_start[1] + 6), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(canvas, (ruler_end[0], ruler_end[1] - 6), (ruler_end[0], ruler_end[1] + 6), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, "1.0 m", (22, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"center x={center[0]:+.2f} y={center[1]:+.2f} z={center[2]:+.2f} m", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"rel depth {center[1] - depth_reference:+.2f} m", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"frame {frame_payload.get('frame_index', 0)}",
        (18, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _fit_frame_to_panel(
    frame: np.ndarray,
    panel_size: Tuple[int, int],
) -> Tuple[np.ndarray, float, int, int]:
    panel_width, panel_height = panel_size
    height, width = frame.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    resized = cv2.resize(
        frame,
        (max(int(width * scale), 1), max(int(height * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((panel_height, panel_width, 3), 18, dtype=np.uint8)
    offset_x = (panel_width - resized.shape[1]) // 2
    offset_y = (panel_height - resized.shape[0]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return canvas, scale, offset_x, offset_y


def _project_camera_point(projection: np.ndarray, point_3d: np.ndarray) -> np.ndarray | None:
    homogeneous = np.append(point_3d, 1.0)
    projected = projection @ homogeneous
    if abs(projected[2]) < 1e-9:
        return None
    return projected[:2] / projected[2]


def _collect_projected_points(
    joints_3d: Dict[str, List[float]],
    projection: np.ndarray | None,
    scale: float,
    offset_x: int,
    offset_y: int,
    panel_width: int,
    panel_height: int,
) -> Dict[str, Tuple[int, int]]:
    if projection is None or not joints_3d:
        return {}
    projected_points: Dict[str, Tuple[int, int]] = {}
    for joint_name, point_3d in joints_3d.items():
        projected = _project_camera_point(projection, np.asarray(point_3d, dtype=float))
        if projected is None:
            continue
        x = int(round(projected[0] * scale + offset_x))
        y = int(round(projected[1] * scale + offset_y))
        if 0 <= x < panel_width and 0 <= y < panel_height:
            projected_points[joint_name] = (x, y)
    return projected_points


def _points_bbox(points: Dict[str, Tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not points:
        return None
    coords = np.asarray(list(points.values()), dtype=float)
    return coords.min(axis=0), coords.max(axis=0)


def _matches_expected_projection(
    projected_points: Dict[str, Tuple[int, int]],
    expected_points: Dict[str, Tuple[int, int]] | None,
) -> bool:
    if not expected_points or len(expected_points) < 6 or len(projected_points) < 6:
        return True

    bbox = _points_bbox(projected_points)
    expected_bbox = _points_bbox(expected_points)
    if bbox is None or expected_bbox is None:
        return True

    min_xy, max_xy = bbox
    expected_min_xy, expected_max_xy = expected_bbox
    expected_size = expected_max_xy - expected_min_xy
    padding = np.maximum(expected_size * 0.45, 28.0)
    expanded_min = expected_min_xy - padding
    expanded_max = expected_max_xy + padding

    inside_hits = 0
    for x, y in projected_points.values():
        if expanded_min[0] <= x <= expanded_max[0] and expanded_min[1] <= y <= expanded_max[1]:
            inside_hits += 1
    if inside_hits < max(4, int(0.45 * len(projected_points))):
        return False

    shared_names = sorted(set(projected_points) & set(expected_points))
    if len(shared_names) >= 4:
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(projected_points[name], dtype=float)
                    - np.asarray(expected_points[name], dtype=float)
                )
            )
            for name in shared_names
        ]
        expected_diag = float(np.linalg.norm(expected_size))
        if np.median(np.asarray(distances, dtype=float)) > max(42.0, 0.6 * expected_diag):
            return False

    center = (min_xy + max_xy) / 2.0
    expected_center = (expected_min_xy + expected_max_xy) / 2.0
    center_distance = float(np.linalg.norm(center - expected_center))
    if center_distance > max(64.0, 0.75 * float(np.linalg.norm(expected_size))):
        return False
    return True


def _draw_projected_pose(
    panel: np.ndarray,
    joints_3d: Dict[str, List[float]],
    projection: np.ndarray | None,
    scale: float,
    offset_x: int,
    offset_y: int,
    background_panel: np.ndarray | None,
) -> np.ndarray:
    if projection is None or not joints_3d:
        return panel

    overlay = panel.copy()
    panel_height, panel_width = panel.shape[:2]
    projected_points = _collect_projected_points(
        joints_3d=joints_3d,
        projection=projection,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        panel_width=panel_width,
        panel_height=panel_height,
    )

    if not _projected_pose_is_reliable(
        projected_points,
        panel_width,
        panel_height,
        panel=panel,
        background_panel=background_panel,
    ):
        return panel

    for joint_a, joint_b in PANEL_CONNECTIONS:
        if joint_a not in projected_points or joint_b not in projected_points:
            continue
        cv2.line(
            overlay,
            projected_points[joint_a],
            projected_points[joint_b],
            (72, 255, 162),
            2,
            cv2.LINE_AA,
        )

    for point in projected_points.values():
        cv2.circle(overlay, point, 4, (45, 210, 255), -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0)


def _extract_landmark_points(
    landmarks: Dict[str, object],
    scale: float,
    offset_x: int,
    offset_y: int,
    panel_width: int,
    panel_height: int,
    confidence_threshold: float,
) -> Dict[str, Tuple[int, int]]:
    points: Dict[str, Tuple[int, int]] = {}
    for joint_name, landmark in landmarks.items():
        score = float(getattr(landmark, "score", 0.0))
        if score < confidence_threshold:
            continue
        x = int(round(float(getattr(landmark, "x")) * scale + offset_x))
        y = int(round(float(getattr(landmark, "y")) * scale + offset_y))
        if 0 <= x < panel_width and 0 <= y < panel_height:
            points[joint_name] = (x, y)
    return points


def _draw_pose_points(
    panel: np.ndarray,
    points: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    if not points:
        return panel
    overlay = panel.copy()
    for joint_a, joint_b in PANEL_CONNECTIONS:
        if joint_a not in points or joint_b not in points:
            continue
        cv2.line(
            overlay,
            points[joint_a],
            points[joint_b],
            (72, 255, 162),
            2,
            cv2.LINE_AA,
        )
    for point in points.values():
        cv2.circle(overlay, point, 4, (45, 210, 255), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.82, panel, 0.18, 0.0)


def _stabilize_pose_points(
    points: Dict[str, Tuple[int, int]],
    expected_points: Dict[str, Tuple[int, int]] | None,
    track_state: Dict[str, object],
) -> Dict[str, Tuple[int, int]]:
    previous_points = {
        name: np.asarray(value, dtype=float)
        for name, value in track_state.get("points", {}).items()
    }
    previous_misses = dict(track_state.get("misses", {}))
    stabilized: Dict[str, Tuple[int, int]] = {}
    next_points: Dict[str, Tuple[float, float]] = {}
    next_misses: Dict[str, int] = {}

    bbox = _points_bbox(expected_points or points)
    if bbox is not None:
        bbox_diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    else:
        bbox_diag = 0.0
    jump_limit = max(POSE_TRACK_MAX_JUMP_PX, 0.22 * bbox_diag)

    all_joint_names = sorted(set(previous_points) | set(points))
    for joint_name in all_joint_names:
        current = points.get(joint_name)
        previous = previous_points.get(joint_name)
        if current is None:
            miss_count = int(previous_misses.get(joint_name, 0)) + 1
            if previous is not None and miss_count <= POSE_TRACK_MAX_MISSES:
                stabilized[joint_name] = tuple(np.round(previous).astype(int))
                next_points[joint_name] = (float(previous[0]), float(previous[1]))
                next_misses[joint_name] = miss_count
            continue

        current_array = np.asarray(current, dtype=float)
        if expected_points and joint_name in expected_points:
            expected_array = np.asarray(expected_points[joint_name], dtype=float)
            current_array = expected_array + 0.6 * (current_array - expected_array)

        if previous is not None:
            delta = current_array - previous
            distance = float(np.linalg.norm(delta))
            if distance > jump_limit:
                current_array = previous + delta / max(distance, 1e-6) * jump_limit
            smoothed = previous * (1.0 - POSE_TRACK_ALPHA) + current_array * POSE_TRACK_ALPHA
        else:
            smoothed = current_array

        stabilized[joint_name] = tuple(np.round(smoothed).astype(int))
        next_points[joint_name] = (float(smoothed[0]), float(smoothed[1]))
        next_misses[joint_name] = 0

    track_state["points"] = next_points
    track_state["misses"] = next_misses
    return stabilized


def _draw_pose_from_landmarks(
    panel: np.ndarray,
    landmarks: Dict[str, object],
    scale: float,
    offset_x: int,
    offset_y: int,
    background_panel: np.ndarray | None,
    confidence_threshold: float,
    expected_points: Dict[str, Tuple[int, int]] | None,
    track_state: Dict[str, object],
) -> np.ndarray:
    if not landmarks:
        track_state["misses"] = {
            name: int(miss_count) + 1
            for name, miss_count in dict(track_state.get("misses", {})).items()
            if int(miss_count) + 1 <= POSE_TRACK_MAX_MISSES
        }
        return panel

    panel_height, panel_width = panel.shape[:2]
    points = _extract_landmark_points(
        landmarks=landmarks,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        panel_width=panel_width,
        panel_height=panel_height,
        confidence_threshold=confidence_threshold,
    )

    if not _projected_pose_is_reliable(
        points,
        panel_width,
        panel_height,
        panel=panel,
        background_panel=background_panel,
    ):
        return panel
    if not _matches_expected_projection(points, expected_points):
        return panel

    stabilized_points = _stabilize_pose_points(
        points=points,
        expected_points=expected_points,
        track_state=track_state,
    )
    if not _projected_pose_is_reliable(
        stabilized_points,
        panel_width,
        panel_height,
        panel=panel,
        background_panel=background_panel,
    ):
        return panel
    return _draw_pose_points(panel, stabilized_points)


def _projected_pose_is_reliable(
    projected_points: Dict[str, Tuple[int, int]],
    panel_width: int,
    panel_height: int,
    panel: np.ndarray,
    background_panel: np.ndarray | None,
) -> bool:
    if len(projected_points) < 6:
        return False
    torso_points = [projected_points[name] for name in PANEL_TORSO_JOINTS if name in projected_points]
    if len(torso_points) < 3:
        return False
    torso_array = np.asarray(torso_points, dtype=float)
    torso_size = torso_array.max(axis=0) - torso_array.min(axis=0)
    points = np.asarray(list(projected_points.values()), dtype=float)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    size = max_xy - min_xy
    area = float(size[0] * size[1])
    panel_area = float(panel_width * panel_height)
    if area < 0.003 * panel_area or area > 0.65 * panel_area:
        return False
    center = points.mean(axis=0)
    if not (-0.05 * panel_width <= center[0] <= 1.05 * panel_width):
        return False
    if not (-0.05 * panel_height <= center[1] <= 1.05 * panel_height):
        return False
    min_dim = float(min(panel_width, panel_height))
    if torso_size.max() < 0.08 * min_dim or torso_size.min() < 0.02 * min_dim:
        return False
    strong_torso_edges = 0
    for joint_a, joint_b, threshold in (
        ("left_shoulder", "right_shoulder", 0.05),
        ("left_hip", "right_hip", 0.04),
        ("left_shoulder", "left_hip", 0.09),
        ("right_shoulder", "right_hip", 0.09),
    ):
        if joint_a not in projected_points or joint_b not in projected_points:
            continue
        point_a = np.asarray(projected_points[joint_a], dtype=float)
        point_b = np.asarray(projected_points[joint_b], dtype=float)
        if float(np.linalg.norm(point_a - point_b)) >= threshold * min_dim:
            strong_torso_edges += 1
    if strong_torso_edges < 2:
        return False
    if background_panel is not None:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.cvtColor(background_panel, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background_gray)
        _, motion_mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(motion_mask, 5)
        point_hits = 0
        for x, y in projected_points.values():
            if 0 <= x < panel_width and 0 <= y < panel_height and motion_mask[y, x] > 0:
                point_hits += 1
        if point_hits < max(3, int(0.35 * len(projected_points))):
            return False
        bbox_pad = 8
        x0 = max(int(min_xy[0]) - bbox_pad, 0)
        y0 = max(int(min_xy[1]) - bbox_pad, 0)
        x1 = min(int(max_xy[0]) + bbox_pad, panel_width)
        y1 = min(int(max_xy[1]) + bbox_pad, panel_height)
        region = motion_mask[y0:y1, x0:x1]
        if region.size == 0 or float(np.count_nonzero(region)) / float(region.size) < 0.02:
            return False
    return True


def _annotate_camera_panel(
    frame: np.ndarray,
    camera_id: str,
    source_frame_index: int,
) -> np.ndarray:
    panel = frame.copy()
    cv2.putText(
        panel,
        camera_id.upper(),
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"src frame {source_frame_index}",
        (16, panel.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return panel


def _advance_capture(capture, num_frames: int) -> None:
    if num_frames <= 0:
        return
    capture.set(cv2.CAP_PROP_POS_FRAMES, num_frames)


def render_multicam_overview(
    config: ProjectConfig,
    chute_name: str,
    reconstruction_path: str | Path,
    output_path: str | Path,
    panel_size: Tuple[int, int] = (480, 320),
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Path:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to render the composite video.")

    reconstruction = _load_reconstruction_payload(reconstruction_path)
    frames = reconstruction.get("frames", [])
    if not frames:
        raise ValueError("The reconstruction file does not contain any frames.")
    frame_lookup = {
        int(frame_payload.get("frame_index", index)): frame_payload
        for index, frame_payload in enumerate(frames)
    }

    sync_offsets = {
        camera_id: int(value)
        for camera_id, value in reconstruction.get("sync_offsets", {}).items()
    }
    chute_dir = Path(config.dataset_root) / chute_name
    camera_videos = discover_camera_videos(chute_dir, config.cameras)
    if len(camera_videos) != len(config.cameras):
        missing = sorted(set(config.cameras) - set(camera_videos))
        raise FileNotFoundError(f"Missing videos for cameras: {missing}")

    calibrations = load_calibration_file(config.calibration_file)
    captures = {}
    fps = None
    projection_mats = {}
    background_panels = {}
    pose_estimators = {}
    pose_track_states = {}
    available_lengths = []
    for camera_id in config.cameras:
        capture = cv2.VideoCapture(str(camera_videos[camera_id]))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for {camera_id}: {camera_videos[camera_id]}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        _advance_capture(capture, max(sync_offsets.get(camera_id, 0), 0))
        captures[camera_id] = capture
        calibration = calibrations.get(camera_id)
        projection_mats[camera_id] = projection_matrix(calibration) if calibration is not None else None
        available_lengths.append(max(total_frames - max(sync_offsets.get(camera_id, 0), 0), 0))
        if fps is None:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        try:
            pose_estimators[camera_id] = create_pose_estimator(
                name=config.pose_backend,
                min_confidence=config.pose_confidence,
            )
        except Exception:
            pose_estimators[camera_id] = None
        pose_track_states[camera_id] = {"points": {}, "misses": {}}

    total_length = min(available_lengths) if available_lengths else 0
    selected_indices = list(range(0, total_length, max(frame_stride, 1)))
    if max_frames is not None:
        selected_indices = selected_indices[:max_frames]
    if not selected_indices:
        raise ValueError("No frames were selected for export.")

    if frame_lookup and max(frame_lookup) + 1 < total_length:
        print(
            "Warning: the reconstruction file is shorter than the synchronized videos. "
            "Later frames will show the camera mosaic without skeleton overlays."
        )

    radius = _global_view_radius(frame_lookup.values())
    valid_centers = [
        _frame_center(frame_payload["joints"])
        for frame_payload in frame_lookup.values()
        if frame_payload.get("joints")
    ]
    depth_reference = float(valid_centers[0][1]) if valid_centers else 0.0
    panel_width, panel_height = panel_size
    output_width = panel_width * 3
    output_height = panel_height * 3
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(frame_stride, 1),
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}.")

    current_source_index = 0
    black_panel = np.full((panel_height, panel_width, 3), 18, dtype=np.uint8)
    skeleton_display_center = None

    try:
        for selected_index in selected_indices:
            target_source_index = selected_index
            step_delta = target_source_index - current_source_index
            for capture in captures.values():
                for _ in range(max(step_delta - 1, 0)):
                    capture.read()
            current_source_index = target_source_index

            mosaic_panels = []
            frame_payload = frame_lookup.get(
                target_source_index,
                {
                    "frame_index": target_source_index,
                    "joints": {},
                    "supporting_cameras": {},
                },
            )
            joints_3d = frame_payload.get("joints", {})
            for camera_id in config.cameras:
                ok, frame = captures[camera_id].read()
                if not ok:
                    mosaic_panels.append(_annotate_camera_panel(black_panel, camera_id, target_source_index + sync_offsets.get(camera_id, 0)))
                    continue
                calibration = calibrations.get(camera_id)
                if calibration is not None:
                    frame = undistort_frame(frame, calibration)
                panel, scale, offset_x, offset_y = _fit_frame_to_panel(frame, panel_size)
                if camera_id not in background_panels:
                    background_panels[camera_id] = panel.copy()
                expected_points = _collect_projected_points(
                    joints_3d=joints_3d,
                    projection=projection_mats.get(camera_id),
                    scale=scale,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    panel_width=panel.shape[1],
                    panel_height=panel.shape[0],
                )
                estimator = pose_estimators.get(camera_id)
                if estimator is not None:
                    pose_frame = estimator.process_frame(
                        frame_bgr=frame,
                        camera_id=camera_id,
                        frame_index=target_source_index + sync_offsets.get(camera_id, 0),
                        timestamp_sec=(target_source_index + sync_offsets.get(camera_id, 0)) / max(fps, 1.0),
                    )
                    panel = _draw_pose_from_landmarks(
                        panel=panel,
                        landmarks=pose_frame.landmarks,
                        scale=scale,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        background_panel=background_panels.get(camera_id),
                        confidence_threshold=config.pose_confidence,
                        expected_points=expected_points,
                        track_state=pose_track_states[camera_id],
                    )
                mosaic_panels.append(
                    _annotate_camera_panel(
                        panel,
                        camera_id=camera_id,
                        source_frame_index=target_source_index + sync_offsets.get(camera_id, 0),
                    )
                )

            if joints_3d:
                current_center = _frame_center(joints_3d)
                if skeleton_display_center is None:
                    skeleton_display_center = current_center
                else:
                    skeleton_display_center = (
                        (1.0 - SKELETON_CENTER_ALPHA) * skeleton_display_center
                        + SKELETON_CENTER_ALPHA * current_center
                    )
            skeleton_panel = _render_skeleton_panel(
                frame_payload,
                panel_size,
                radius,
                depth_reference=depth_reference,
                display_center=skeleton_display_center,
            )
            all_panels = mosaic_panels + [skeleton_panel]
            rows = []
            for row_index in range(3):
                row_panels = all_panels[row_index * 3 : (row_index + 1) * 3]
                rows.append(np.hstack(row_panels))
            writer.write(np.vstack(rows))
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()

    return output
