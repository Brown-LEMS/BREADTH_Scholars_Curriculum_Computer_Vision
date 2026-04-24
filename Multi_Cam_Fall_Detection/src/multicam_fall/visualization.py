from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .ground_truth import label_for_frame, load_ground_truth_json


POSE_CONNECTIONS = [
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_thumb"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_thumb"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),
    ("nose", "left_eye_inner"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("left_eye_outer", "left_ear"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
]


def _load_reconstruction_frames(path: str | Path) -> Tuple[str, List[Dict[str, object]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    valid_frames = [frame for frame in payload.get("frames", []) if frame.get("joints")]
    return payload.get("chute", "unknown"), valid_frames


def _axis_limits(frames: Iterable[Dict[str, object]]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    points = []
    for frame in frames:
        for value in frame.get("joints", {}).values():
            points.append(value)
    coords = np.asarray(points, dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def _frame_center(joints: Dict[str, List[float]]) -> np.ndarray:
    torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    anchors = [joints[name] for name in torso_names if name in joints]
    if len(anchors) >= 2:
        return np.asarray(anchors, dtype=float).mean(axis=0)
    return np.asarray(list(joints.values()), dtype=float).mean(axis=0)


def _view_radius(frames: Iterable[Dict[str, object]]) -> float:
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


def save_reconstruction_gif(
    reconstruction_path: str | Path,
    output_path: str | Path,
    fps: int = 10,
    frame_stride: int = 2,
    ground_truth_path: str | Path | None = None,
) -> Path:
    chute_name, frames = _load_reconstruction_frames(reconstruction_path)
    if not frames:
        raise ValueError(
            "No 3D joints were found in the reconstruction file. "
            "Re-run the reconstruction step with looser triangulation thresholds "
            "or a different pose backend before visualizing."
        )

    selected_frames = frames[:: max(frame_stride, 1)]
    view_radius = _view_radius(selected_frames)
    ground_truth = load_ground_truth_json(ground_truth_path) if ground_truth_path else None

    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection="3d")

    def update(frame_position: int):
        axis.cla()
        frame = selected_frames[frame_position]
        joints = frame["joints"]
        points = np.asarray(list(joints.values()), dtype=float)
        center = _frame_center(joints)

        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c="#d94f4f", s=28)
        for joint_a, joint_b in POSE_CONNECTIONS:
            if joint_a not in joints or joint_b not in joints:
                continue
            segment = np.asarray([joints[joint_a], joints[joint_b]], dtype=float)
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], c="#2f6fed", linewidth=2)

        axis.set_xlim(center[0] - view_radius, center[0] + view_radius)
        axis.set_ylim(center[1] - view_radius, center[1] + view_radius)
        axis.set_zlim(center[2] - view_radius, center[2] + view_radius)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.view_init(elev=18, azim=-64)
        axis.set_title(f"{chute_name} - frame {frame['frame_index']}")

        if ground_truth:
            label = label_for_frame(ground_truth, chute_name, frame["frame_index"])
            if label:
                axis.text2D(
                    0.03,
                    0.96,
                    f"GT: {label['description']}",
                    transform=axis.transAxes,
                    fontsize=11,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

    animation = FuncAnimation(figure, update, frames=len(selected_frames), interval=1000 / fps)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output
