from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .models import Pose2DFrame


def _joint_value(frame: Pose2DFrame, joint_name: str) -> float:
    if joint_name == "mid_shoulder":
        left = frame.landmarks.get("left_shoulder")
        right = frame.landmarks.get("right_shoulder")
        if left and right:
            return (left.y + right.y) / 2.0
    if joint_name == "mid_hip":
        left = frame.landmarks.get("left_hip")
        right = frame.landmarks.get("right_hip")
        if left and right:
            return (left.y + right.y) / 2.0
    landmark = frame.landmarks.get(joint_name)
    if landmark:
        return landmark.y
    return np.nan


def build_motion_signal(frames: Iterable[Pose2DFrame], joint_name: str) -> np.ndarray:
    coords = np.asarray([_joint_value(frame, joint_name) for frame in frames], dtype=float)
    if len(coords) == 0:
        return coords
    valid = np.isfinite(coords)
    if valid.any():
        coords = np.interp(
            np.arange(len(coords)),
            np.flatnonzero(valid),
            coords[valid],
        )
    else:
        coords = np.zeros_like(coords)
    signal = np.diff(coords, prepend=coords[0])
    signal -= signal.mean() if len(signal) else 0.0
    std = signal.std()
    return signal / std if std > 1e-6 else signal


def estimate_frame_offset(reference: np.ndarray, target: np.ndarray, max_offset: int) -> int:
    if len(reference) == 0 or len(target) == 0:
        return 0
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_offset, max_offset + 1):
        if shift >= 0:
            ref_slice = reference[shift:]
            tgt_slice = target[: len(ref_slice)]
        else:
            tgt_slice = target[-shift:]
            ref_slice = reference[: len(tgt_slice)]
        if len(ref_slice) < 10 or len(tgt_slice) < 10:
            continue
        score = float(np.dot(ref_slice, tgt_slice) / len(ref_slice))
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_shift


def estimate_delays(
    pose_sequences: Dict[str, List[Pose2DFrame]],
    reference_camera: str,
    joint_name: str,
    max_offset: int,
) -> Dict[str, int]:
    if reference_camera not in pose_sequences:
        raise ValueError(f"Reference camera `{reference_camera}` was not found in pose sequences.")

    reference_signal = build_motion_signal(pose_sequences[reference_camera], joint_name)
    offsets = {reference_camera: 0}
    for camera_id, frames in pose_sequences.items():
        if camera_id == reference_camera:
            continue
        target_signal = build_motion_signal(frames, joint_name)
        offsets[camera_id] = estimate_frame_offset(reference_signal, target_signal, max_offset)
    return offsets


def apply_delays_to_sequences(
    pose_sequences: Dict[str, List[Pose2DFrame]],
    delays: Dict[str, int],
) -> Dict[str, List[Pose2DFrame]]:
    if not pose_sequences:
        return {}
    baseline = min(delays.values(), default=0)
    aligned: Dict[str, List[Pose2DFrame]] = {}
    for camera_id, frames in pose_sequences.items():
        delay = delays.get(camera_id, 0) - baseline
        aligned[camera_id] = frames[delay:] if delay > 0 else frames[:]
    min_length = min((len(frames) for frames in aligned.values()), default=0)
    return {camera_id: frames[:min_length] for camera_id, frames in aligned.items()}
