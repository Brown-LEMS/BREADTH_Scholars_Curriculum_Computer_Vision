from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .calibration import (
    fundamental_from_calibration,
    load_calibration_file,
    load_sequence_delays,
    undistort_frame,
)
from .config import ProjectConfig
from .dataset import discover_camera_videos
from .features import estimate_fundamental_matrix
from .models import Pose2DFrame
from .postprocess import refine_reconstructions
from .pose2d import create_pose_estimator
from .synchronization import apply_delays_to_sequences, estimate_delays
from .triangulation import reconstruct_frame

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


class MultiCameraPipeline:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.calibrations = load_calibration_file(config.calibration_file)
        self.sequence_delays = load_sequence_delays(config.calibration_file)

    def run(self, chute_name: str | None = None) -> Path:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to read AVI videos. Install it with `pip install opencv-contrib-python`."
            )

        chute = chute_name or self.config.default_chute
        chute_dir = Path(self.config.dataset_root) / chute
        if not chute_dir.exists():
            raise FileNotFoundError(f"Could not find chute directory: {chute_dir}")

        camera_videos = discover_camera_videos(chute_dir, self.config.cameras)
        if not camera_videos:
            raise FileNotFoundError(f"No camera videos were found under {chute_dir}")

        pose_estimator = create_pose_estimator(
            name=self.config.pose_backend,
            min_confidence=self.config.pose_confidence,
        )

        pose_sequences, preview_frames = self._extract_pose_sequences(camera_videos, pose_estimator)
        use_report_delays = (
            self.config.prefer_sequence_delays
            and chute in self.sequence_delays
            and all(camera_id in self.sequence_delays[chute] for camera_id in camera_videos)
        )

        if use_report_delays:
            sync_offsets = {
                camera_id: self.sequence_delays[chute][camera_id]
                for camera_id in camera_videos
            }
        else:
            sync_offsets = estimate_delays(
                pose_sequences=pose_sequences,
                reference_camera=self.config.sync_reference_camera,
                joint_name=self.config.sync_signal_joint,
                max_offset=self.config.sync_search_radius,
            )
        for camera_id, calibration in self.calibrations.items():
            sync_offsets[camera_id] = sync_offsets.get(camera_id, 0) + calibration.frame_delay

        aligned_sequences = apply_delays_to_sequences(pose_sequences, sync_offsets)
        pairwise_fundamentals = self._build_pairwise_fundamentals(preview_frames)
        reconstructions = self._reconstruct(aligned_sequences, pairwise_fundamentals)
        if self.config.enable_postprocess:
            reconstructions = refine_reconstructions(
                reconstructions,
                max_interpolation_gap=self.config.interpolation_gap_frames,
            )
        output_path = self._write_output(chute, sync_offsets, reconstructions)
        return output_path

    def _extract_pose_sequences(
        self,
        camera_videos: Dict[str, Path],
        pose_estimator,
    ) -> Tuple[Dict[str, List[Pose2DFrame]], Dict[str, np.ndarray]]:
        pose_sequences: Dict[str, List[Pose2DFrame]] = {}
        preview_frames: Dict[str, np.ndarray] = {}

        for camera_id, video_path in camera_videos.items():
            calibration = self.calibrations.get(camera_id)
            capture = cv2.VideoCapture(str(video_path))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_cursor = 0
            sampled_frames = 0
            frames: List[Pose2DFrame] = []

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_cursor % self.config.sample_every_n_frames != 0:
                    frame_cursor += 1
                    continue
                processed = undistort_frame(frame, calibration) if calibration else frame
                if camera_id not in preview_frames:
                    preview_frames[camera_id] = processed.copy()
                timestamp_sec = frame_cursor / fps
                frames.append(
                    pose_estimator.process_frame(
                        frame_bgr=processed,
                        camera_id=camera_id,
                        frame_index=frame_cursor,
                        timestamp_sec=timestamp_sec,
                    )
                )
                frame_cursor += 1
                sampled_frames += 1
                if self.config.max_frames and sampled_frames >= self.config.max_frames:
                    break

            capture.release()
            pose_sequences[camera_id] = frames

        return pose_sequences, preview_frames

    def _build_pairwise_fundamentals(
        self,
        preview_frames: Dict[str, np.ndarray],
    ) -> Dict[Tuple[str, str], np.ndarray]:
        fundamentals: Dict[Tuple[str, str], np.ndarray] = {}
        for camera_a, camera_b in combinations(sorted(preview_frames), 2):
            fundamental = None
            if self.config.use_calibration_fundamental_matrix:
                fundamental = fundamental_from_calibration(
                    self.calibrations.get(camera_a),
                    self.calibrations.get(camera_b),
                )
            if fundamental is None and self.config.enable_feature_fundamental_matrix:
                fundamental = estimate_fundamental_matrix(
                    preview_frames[camera_a],
                    preview_frames[camera_b],
                )
            if fundamental is not None:
                fundamentals[(camera_a, camera_b)] = fundamental
                fundamentals[(camera_b, camera_a)] = fundamental.T
        return fundamentals

    def _reconstruct(
        self,
        aligned_sequences: Dict[str, List[Pose2DFrame]],
        pairwise_fundamentals: Dict[Tuple[str, str], np.ndarray],
    ):
        min_length = min((len(frames) for frames in aligned_sequences.values()), default=0)
        reconstructions = []
        for step in range(min_length):
            per_camera_landmarks = {}
            timestamp_sec = 0.0
            for camera_id, frames in aligned_sequences.items():
                frame = frames[step]
                timestamp_sec = frame.timestamp_sec
                per_camera_landmarks[camera_id] = {
                    name: (landmark.as_array(), landmark.score)
                    for name, landmark in frame.landmarks.items()
                    if landmark.score >= self.config.pose_confidence
                }
            reconstructions.append(
                reconstruct_frame(
                    frame_index=step,
                    timestamp_sec=timestamp_sec,
                    per_camera_landmarks=per_camera_landmarks,
                    calibrations=self.calibrations,
                    pairwise_fundamentals=pairwise_fundamentals,
                    min_views=self.config.min_views_triangulation,
                    epipolar_threshold_px=self.config.epipolar_threshold_px,
                    reprojection_threshold_px=self.config.reprojection_threshold_px,
                    min_triangulation_angle_deg=self.config.min_triangulation_angle_deg,
                )
            )
        return reconstructions

    def _write_output(self, chute: str, sync_offsets: Dict[str, int], reconstructions) -> Path:
        output_dir = Path(self.config.outputs_root) / chute
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.config.output_json_name
        payload = {
            "chute": chute,
            "sync_offsets": sync_offsets,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "joints": frame.joints,
                    "supporting_cameras": frame.supporting_cameras,
                }
                for frame in reconstructions
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
