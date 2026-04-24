from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .models import Reconstruction3DFrame


BONE_SPECS: List[Tuple[str, str, str, float]] = [
    ("left_shoulder", "right_shoulder", "torso", 1.4),
    ("left_hip", "right_hip", "torso", 1.4),
    ("left_shoulder", "left_hip", "torso", 1.8),
    ("right_shoulder", "right_hip", "torso", 1.8),
    ("left_shoulder", "left_elbow", "limb", 1.0),
    ("left_elbow", "left_wrist", "limb", 1.0),
    ("right_shoulder", "right_elbow", "limb", 1.0),
    ("right_elbow", "right_wrist", "limb", 1.0),
    ("left_hip", "left_knee", "limb", 1.1),
    ("left_knee", "left_ankle", "limb", 1.1),
    ("right_hip", "right_knee", "limb", 1.1),
    ("right_knee", "right_ankle", "limb", 1.1),
    ("left_ankle", "left_heel", "leaf", 0.8),
    ("left_heel", "left_foot_index", "leaf", 0.8),
    ("right_ankle", "right_heel", "leaf", 0.8),
    ("right_heel", "right_foot_index", "leaf", 0.8),
    ("nose", "left_shoulder", "head", 0.9),
    ("nose", "right_shoulder", "head", 0.9),
    ("left_shoulder", "left_ear", "head", 0.8),
    ("right_shoulder", "right_ear", "head", 0.8),
]

TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
CORE_JOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
MIN_RELIABLE_CORE_JOINTS = 6
MIN_RELIABLE_TORSO_JOINTS = 3
MIN_PRESENCE_RUN_FRAMES = 20
REMOVAL_PRIORITY = {
    "left_foot_index": 8,
    "right_foot_index": 8,
    "left_heel": 8,
    "right_heel": 8,
    "left_ankle": 7,
    "right_ankle": 7,
    "left_knee": 6,
    "right_knee": 6,
    "left_wrist": 6,
    "right_wrist": 6,
    "left_elbow": 5,
    "right_elbow": 5,
    "left_ear": 4,
    "right_ear": 4,
    "nose": 4,
    "left_shoulder": 3,
    "right_shoulder": 3,
    "left_hip": 2,
    "right_hip": 2,
}


def _distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float)))


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm <= 1e-9:
            return np.asarray([1.0, 0.0, 0.0], dtype=float)
        return fallback / fallback_norm
    return vector / norm


def _median_without_outliers(values: Iterable[float]) -> float | None:
    data = np.asarray(list(values), dtype=float)
    if data.size < 5:
        return None
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    if mad > 1e-9:
        robust_scale = 1.4826 * mad
        data = data[np.abs(data - median) <= 3.0 * robust_scale]
    if data.size == 0:
        return None
    return float(np.median(data))


def estimate_bone_lengths(frames: Sequence[Reconstruction3DFrame]) -> Dict[Tuple[str, str], float]:
    lengths: Dict[Tuple[str, str], List[float]] = {}
    for joint_a, joint_b, _, _ in BONE_SPECS:
        lengths[(joint_a, joint_b)] = []

    for frame in frames:
        for joint_a, joint_b, _, _ in BONE_SPECS:
            if joint_a not in frame.joints or joint_b not in frame.joints:
                continue
            lengths[(joint_a, joint_b)].append(_distance(frame.joints[joint_a], frame.joints[joint_b]))

    expected = {}
    for key, values in lengths.items():
        median = _median_without_outliers(values)
        if median is not None:
            expected[key] = median
    return expected


def _bone_ratio_bounds(group: str) -> Tuple[float, float]:
    if group == "torso":
        return 0.55, 1.8
    if group == "head":
        return 0.45, 1.9
    if group == "leaf":
        return 0.3, 2.0
    return 0.35, 2.1


def _body_center(joints: Dict[str, List[float]]) -> np.ndarray | None:
    anchors = [np.asarray(joints[name], dtype=float) for name in TORSO_JOINTS if name in joints]
    if len(anchors) >= 2:
        return np.mean(anchors, axis=0)
    if not joints:
        return None
    return np.mean(np.asarray(list(joints.values()), dtype=float), axis=0)


def _mean_present(joints: Dict[str, np.ndarray], joint_names: Sequence[str]) -> np.ndarray | None:
    points = [joints[name] for name in joint_names if name in joints]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _torso_joint_count(joints: Dict[str, np.ndarray] | Dict[str, List[float]]) -> int:
    return sum(1 for name in TORSO_JOINTS if name in joints)


def _has_reliable_observation(observed: Dict[str, np.ndarray]) -> bool:
    return len(observed) >= MIN_RELIABLE_CORE_JOINTS and _torso_joint_count(observed) >= MIN_RELIABLE_TORSO_JOINTS


def _frame_has_person(joints: Dict[str, List[float]]) -> bool:
    if len(joints) < 8 or _torso_joint_count(joints) < 3:
        return False
    try:
        shoulder_center = _mean_present(
            {name: np.asarray(value, dtype=float) for name, value in joints.items()},
            ("left_shoulder", "right_shoulder"),
        )
        hip_center = _mean_present(
            {name: np.asarray(value, dtype=float) for name, value in joints.items()},
            ("left_hip", "right_hip"),
        )
        if shoulder_center is None or hip_center is None:
            return False
        torso_length = float(np.linalg.norm(shoulder_center - hip_center))
        if not (0.35 <= torso_length <= 3.5):
            return False
    except Exception:
        return False
    return True


def _torso_scale(joints: Dict[str, List[float]], expected_lengths: Dict[Tuple[str, str], float]) -> float:
    candidates = []
    for key in (
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ):
        if key in expected_lengths:
            candidates.append(expected_lengths[key])
    if candidates:
        return float(np.median(np.asarray(candidates, dtype=float)))
    center = _body_center(joints)
    if center is None or not joints:
        return 1.0
    distances = [float(np.linalg.norm(np.asarray(point, dtype=float) - center)) for point in joints.values()]
    return max(float(np.median(np.asarray(distances, dtype=float))), 1.0)


def _template_lengths(expected_lengths: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    torso_height = float(
        np.median(
            np.asarray(
                [
                    expected_lengths.get(("left_shoulder", "left_hip"), 1.55),
                    expected_lengths.get(("right_shoulder", "right_hip"), 1.55),
                ],
                dtype=float,
            )
        )
    )
    shoulder_width = expected_lengths.get(("left_shoulder", "right_shoulder"), torso_height * 0.3)
    hip_width = expected_lengths.get(("left_hip", "right_hip"), torso_height * 0.22)
    return {
        "torso_height": torso_height,
        "shoulder_width": float(min(max(shoulder_width, torso_height * 0.25), torso_height * 0.45)),
        "hip_width": float(min(max(hip_width, torso_height * 0.18), torso_height * 0.35)),
        "upper_arm": torso_height * 0.55,
        "lower_arm": torso_height * 0.5,
        "thigh": torso_height * 0.65,
        "shank": torso_height * 0.65,
        "head_height": torso_height * 0.3,
    }


def cleanup_frame(
    frame: Reconstruction3DFrame,
    expected_lengths: Dict[Tuple[str, str], float],
) -> Reconstruction3DFrame:
    joints = {name: list(value) for name, value in frame.joints.items()}
    supporting = {name: list(value) for name, value in frame.supporting_cameras.items()}
    if not joints:
        return replace(frame)

    body_center = _body_center(joints)
    torso_scale = _torso_scale(joints, expected_lengths)
    if body_center is not None:
        for joint_name in list(joints):
            point = np.asarray(joints[joint_name], dtype=float)
            if float(np.linalg.norm(point - body_center)) > 3.2 * torso_scale:
                joints.pop(joint_name, None)
                supporting.pop(joint_name, None)

    max_iterations = len(joints)
    for _ in range(max_iterations):
        penalties: Dict[str, float] = {}
        for joint_a, joint_b, group, weight in BONE_SPECS:
            expected = expected_lengths.get((joint_a, joint_b))
            if expected is None or joint_a not in joints or joint_b not in joints:
                continue
            length = _distance(joints[joint_a], joints[joint_b])
            ratio = length / max(expected, 1e-6)
            min_ratio, max_ratio = _bone_ratio_bounds(group)
            if min_ratio <= ratio <= max_ratio:
                continue
            severity = max(min_ratio / max(ratio, 1e-6), ratio / max_ratio) - 1.0
            penalties[joint_a] = penalties.get(joint_a, 0.0) + severity * weight
            penalties[joint_b] = penalties.get(joint_b, 0.0) + severity * weight

        if not penalties:
            break

        worst_joint = max(
            penalties,
            key=lambda joint_name: (
                penalties[joint_name],
                1.0 if len(supporting.get(joint_name, [])) <= 2 else 0.0,
                REMOVAL_PRIORITY.get(joint_name, 0),
            ),
        )
        if penalties[worst_joint] < 0.45:
            break
        joints.pop(worst_joint, None)
        supporting.pop(worst_joint, None)
        if len(joints) < 4:
            break

    return Reconstruction3DFrame(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        joints=joints,
        supporting_cameras=supporting,
    )


def _interpolate_short_gaps(series: np.ndarray, max_gap: int) -> np.ndarray:
    result = series.copy()
    valid = ~np.isnan(result)
    if valid.sum() < 2:
        return result
    index = 0
    length = len(result)
    while index < length:
        if not np.isnan(result[index]):
            index += 1
            continue
        start = index
        while index < length and np.isnan(result[index]):
            index += 1
        end = index
        gap = end - start
        if start == 0 or end >= length or gap > max_gap:
            continue
        left = result[start - 1]
        right = result[end]
        step = (right - left) / (gap + 1)
        for offset in range(gap):
            result[start + offset] = left + step * (offset + 1)
    return result


def _moving_average(series: np.ndarray, window: int = 5) -> np.ndarray:
    result = series.copy()
    radius = window // 2
    for index in range(len(series)):
        if np.isnan(series[index]):
            continue
        start = max(0, index - radius)
        end = min(len(series), index + radius + 1)
        neighborhood = series[start:end]
        valid = neighborhood[~np.isnan(neighborhood)]
        if valid.size >= 2:
            result[index] = float(np.mean(valid))
    return result


def smooth_frames(
    frames: Sequence[Reconstruction3DFrame],
    max_interpolation_gap: int,
) -> List[Reconstruction3DFrame]:
    if not frames:
        return []

    joint_names = sorted({joint_name for frame in frames for joint_name in frame.joints})
    smoothed_frames = [
        Reconstruction3DFrame(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            joints={name: list(value) for name, value in frame.joints.items()},
            supporting_cameras={name: list(value) for name, value in frame.supporting_cameras.items()},
        )
        for frame in frames
    ]

    for joint_name in joint_names:
        coords = np.full((len(frames), 3), np.nan, dtype=float)
        existing = np.zeros(len(frames), dtype=bool)
        for index, frame in enumerate(frames):
            if joint_name in frame.joints:
                coords[index] = np.asarray(frame.joints[joint_name], dtype=float)
                existing[index] = True

        if existing.sum() < 2:
            continue

        for axis in range(3):
            coords[:, axis] = _interpolate_short_gaps(coords[:, axis], max_interpolation_gap)
            coords[:, axis] = _moving_average(coords[:, axis], window=5)

        for index, frame in enumerate(smoothed_frames):
            if np.isnan(coords[index]).any():
                continue
            frame.joints[joint_name] = coords[index].astype(float).tolist()
            if joint_name not in frame.supporting_cameras:
                frame.supporting_cameras[joint_name] = ["postprocess"]

    return smoothed_frames


def _pick_direction(
    parent: np.ndarray,
    observed_targets: Sequence[np.ndarray | None],
    previous_direction: np.ndarray | None,
    default_direction: np.ndarray,
    upward_axis: np.ndarray | None = None,
) -> np.ndarray:
    direction = None
    for target in observed_targets:
        if target is None:
            continue
        vector = target - parent
        if float(np.linalg.norm(vector)) > 1e-6:
            direction = vector
            break
    if direction is None and previous_direction is not None:
        direction = previous_direction
    if direction is None:
        direction = default_direction
    direction = np.asarray(direction, dtype=float)
    if upward_axis is not None and float(np.dot(direction, upward_axis)) > 0.25:
        direction = direction - upward_axis * float(np.dot(direction, upward_axis))
    return _unit(direction, np.asarray(default_direction, dtype=float))


def fit_humanoid_skeleton(
    frames: Sequence[Reconstruction3DFrame],
    expected_lengths: Dict[Tuple[str, str], float],
    max_carry_gap: int,
) -> List[Reconstruction3DFrame]:
    template = _template_lengths(expected_lengths)
    fitted_frames: List[Reconstruction3DFrame] = []
    previous_state: Dict[str, np.ndarray] | None = None

    for frame in frames:
        observed = {
            name: np.asarray(value, dtype=float)
            for name, value in frame.joints.items()
            if name in CORE_JOINTS
        }
        reliable_observation = _has_reliable_observation(observed)
        pelvis_observed = _mean_present(observed, ("left_hip", "right_hip"))
        shoulder_observed = _mean_present(observed, ("left_shoulder", "right_shoulder"))

        if not reliable_observation and previous_state is None:
            fitted_frames.append(
                Reconstruction3DFrame(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                )
            )
            continue
        if not reliable_observation and previous_state is not None:
            carry_gap = int(previous_state.get("carry_gap", 0)) + 1
            if carry_gap > max_carry_gap:
                previous_state = None
                fitted_frames.append(
                    Reconstruction3DFrame(
                        frame_index=frame.frame_index,
                        timestamp_sec=frame.timestamp_sec,
                    )
                )
                continue
        else:
            carry_gap = 0

        torso_direction = (
            shoulder_observed - pelvis_observed
            if pelvis_observed is not None and shoulder_observed is not None
            else previous_state["torso_direction"] if previous_state is not None else np.asarray([0.0, 0.0, 1.0])
        )
        torso_direction = _unit(torso_direction, np.asarray([0.0, 0.0, 1.0]))

        lateral_source = None
        if "left_shoulder" in observed and "right_shoulder" in observed:
            lateral_source = observed["right_shoulder"] - observed["left_shoulder"]
        elif "left_hip" in observed and "right_hip" in observed:
            lateral_source = observed["right_hip"] - observed["left_hip"]
        elif previous_state is not None:
            lateral_source = previous_state["lateral_direction"]
        else:
            lateral_source = np.asarray([1.0, 0.0, 0.0])
        lateral_source = np.asarray(lateral_source, dtype=float)
        lateral_source = lateral_source - torso_direction * float(np.dot(lateral_source, torso_direction))
        lateral_direction = _unit(lateral_source, np.asarray([1.0, 0.0, 0.0]))
        if previous_state is not None and float(np.dot(lateral_direction, previous_state["lateral_direction"])) < 0:
            lateral_direction = -lateral_direction
        if previous_state is not None:
            torso_direction = _unit(
                0.7 * torso_direction + 0.3 * previous_state["torso_direction"],
                previous_state["torso_direction"],
            )
            lateral_direction = _unit(
                0.7 * lateral_direction + 0.3 * previous_state["lateral_direction"],
                previous_state["lateral_direction"],
            )
        forward_direction = _unit(
            np.cross(lateral_direction, torso_direction),
            previous_state["forward_direction"] if previous_state is not None else np.asarray([0.0, 1.0, 0.0]),
        )

        pelvis_center = pelvis_observed
        shoulder_center = shoulder_observed
        if pelvis_center is None and shoulder_center is not None:
            pelvis_center = shoulder_center - torso_direction * template["torso_height"]
        if shoulder_center is None and pelvis_center is not None:
            shoulder_center = pelvis_center + torso_direction * template["torso_height"]
        if pelvis_center is None and shoulder_center is None and previous_state is not None:
            pelvis_center = previous_state["pelvis_center"]
            shoulder_center = previous_state["shoulder_center"]

        if pelvis_center is None or shoulder_center is None:
            fitted_frames.append(
                Reconstruction3DFrame(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                )
            )
            continue

        center_mid = (pelvis_center + shoulder_center) / 2.0
        if float(np.linalg.norm(shoulder_center - pelvis_center)) < 0.6 * template["torso_height"] or float(
            np.linalg.norm(shoulder_center - pelvis_center)
        ) > 1.5 * template["torso_height"]:
            pelvis_center = center_mid - torso_direction * (template["torso_height"] / 2.0)
            shoulder_center = center_mid + torso_direction * (template["torso_height"] / 2.0)
        if previous_state is not None:
            pelvis_center = 0.75 * pelvis_center + 0.25 * previous_state["pelvis_center"]
            shoulder_center = 0.75 * shoulder_center + 0.25 * previous_state["shoulder_center"]

        fitted_joints = {
            "left_hip": (pelvis_center - lateral_direction * (template["hip_width"] / 2.0)).tolist(),
            "right_hip": (pelvis_center + lateral_direction * (template["hip_width"] / 2.0)).tolist(),
            "left_shoulder": (shoulder_center - lateral_direction * (template["shoulder_width"] / 2.0)).tolist(),
            "right_shoulder": (shoulder_center + lateral_direction * (template["shoulder_width"] / 2.0)).tolist(),
        }

        previous_directions = previous_state.get("directions", {}) if previous_state is not None else {}

        for side, lateral_sign in (("left", -1.0), ("right", 1.0)):
            shoulder = np.asarray(fitted_joints[f"{side}_shoulder"], dtype=float)
            hip = np.asarray(fitted_joints[f"{side}_hip"], dtype=float)

            upper_arm_direction = _pick_direction(
                shoulder,
                [observed.get(f"{side}_elbow"), observed.get(f"{side}_wrist")],
                previous_directions.get(f"{side}_upper_arm"),
                lateral_direction * lateral_sign - 0.25 * torso_direction,
            )
            elbow = shoulder + upper_arm_direction * template["upper_arm"]
            lower_arm_direction = _pick_direction(
                elbow,
                [observed.get(f"{side}_wrist")],
                previous_directions.get(f"{side}_lower_arm"),
                upper_arm_direction,
            )
            wrist = elbow + lower_arm_direction * template["lower_arm"]

            thigh_direction = _pick_direction(
                hip,
                [observed.get(f"{side}_knee"), observed.get(f"{side}_ankle")],
                previous_directions.get(f"{side}_thigh"),
                -torso_direction + 0.1 * forward_direction,
                upward_axis=torso_direction,
            )
            knee = hip + thigh_direction * template["thigh"]
            shank_direction = _pick_direction(
                knee,
                [observed.get(f"{side}_ankle")],
                previous_directions.get(f"{side}_shank"),
                thigh_direction,
                upward_axis=torso_direction,
            )
            ankle = knee + shank_direction * template["shank"]

            fitted_joints[f"{side}_elbow"] = elbow.tolist()
            fitted_joints[f"{side}_wrist"] = wrist.tolist()
            fitted_joints[f"{side}_knee"] = knee.tolist()
            fitted_joints[f"{side}_ankle"] = ankle.tolist()

            previous_directions[f"{side}_upper_arm"] = upper_arm_direction
            previous_directions[f"{side}_lower_arm"] = lower_arm_direction
            previous_directions[f"{side}_thigh"] = thigh_direction
            previous_directions[f"{side}_shank"] = shank_direction

        head_direction = _pick_direction(
            shoulder_center,
            [observed.get("nose")],
            previous_directions.get("head"),
            torso_direction,
        )
        if float(np.dot(head_direction, torso_direction)) < 0.5:
            head_direction = torso_direction
        fitted_joints["nose"] = (shoulder_center + head_direction * template["head_height"]).tolist()
        previous_directions["head"] = head_direction

        previous_state = {
            "pelvis_center": pelvis_center,
            "shoulder_center": shoulder_center,
            "torso_direction": torso_direction,
            "lateral_direction": lateral_direction,
            "forward_direction": forward_direction,
            "directions": previous_directions,
            "carry_gap": carry_gap,
        }

        fitted_frames.append(
            Reconstruction3DFrame(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                joints=fitted_joints,
                supporting_cameras={joint_name: ["postprocess_fit"] for joint_name in fitted_joints},
            )
        )

    return fitted_frames


def suppress_short_presence_runs(
    frames: Sequence[Reconstruction3DFrame],
    min_run_length: int,
) -> List[Reconstruction3DFrame]:
    if not frames:
        return []

    suppressed = [
        Reconstruction3DFrame(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            joints={name: list(value) for name, value in frame.joints.items()},
            supporting_cameras={name: list(value) for name, value in frame.supporting_cameras.items()},
        )
        for frame in frames
    ]
    flags = [_frame_has_person(frame.joints) for frame in suppressed]

    start = None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
            continue
        if (not flag) and start is not None:
            if index - start < min_run_length:
                for run_index in range(start, index):
                    suppressed[run_index].joints.clear()
                    suppressed[run_index].supporting_cameras.clear()
            start = None
    return suppressed


def refine_reconstructions(
    frames: Sequence[Reconstruction3DFrame],
    max_interpolation_gap: int,
) -> List[Reconstruction3DFrame]:
    expected_lengths = estimate_bone_lengths(frames)
    cleaned = [cleanup_frame(frame, expected_lengths) for frame in frames]
    smoothed = smooth_frames(cleaned, max_interpolation_gap=max_interpolation_gap)
    refreshed_lengths = estimate_bone_lengths(smoothed)
    normalized = fit_humanoid_skeleton(
        smoothed,
        refreshed_lengths,
        max_carry_gap=max_interpolation_gap,
    )
    normalized_lengths = estimate_bone_lengths(normalized)
    final_frames = [cleanup_frame(frame, normalized_lengths) for frame in normalized]
    stabilized_final = smooth_frames(
        final_frames,
        max_interpolation_gap=max(max_interpolation_gap, 5),
    )
    stabilized_lengths = estimate_bone_lengths(stabilized_final)
    stabilized_final = [cleanup_frame(frame, stabilized_lengths) for frame in stabilized_final]
    return suppress_short_presence_runs(stabilized_final, min_run_length=MIN_PRESENCE_RUN_FRAMES)
