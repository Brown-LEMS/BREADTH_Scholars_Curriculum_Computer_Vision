from __future__ import annotations

from typing import Dict

import numpy as np

from .models import Landmark2D, Pose2DFrame

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime
    mp = None


class BasePoseEstimator:
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        frame_index: int,
        timestamp_sec: float,
    ) -> Pose2DFrame:
        raise NotImplementedError


class MediaPipePoseEstimator(BasePoseEstimator):
    def __init__(self, min_confidence: float = 0.5) -> None:
        if mp is None or cv2 is None:
            raise RuntimeError(
                "MediaPipe pose requires `mediapipe` and `opencv-contrib-python`."
            )
        self._pose_module = mp.solutions.pose
        try:
            self._pose = self._pose_module.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "MediaPipe Pose failed to initialize. On macOS headless or sandboxed runs, "
                "this can happen because NSOpenGLPixelFormat cannot be created. "
                "If this happens on your machine, switch to another 2D pose backend or run "
                "the pipeline in a full desktop Python environment."
            ) from exc
        self._landmark_names = [
            landmark.name.lower() for landmark in self._pose_module.PoseLandmark
        ]

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        frame_index: int,
        timestamp_sec: float,
    ) -> Pose2DFrame:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        height, width = frame_bgr.shape[:2]
        landmarks: Dict[str, Landmark2D] = {}

        if result.pose_landmarks:
            for name, landmark in zip(self._landmark_names, result.pose_landmarks.landmark):
                landmarks[name] = Landmark2D(
                    name=name,
                    x=float(landmark.x * width),
                    y=float(landmark.y * height),
                    score=float(getattr(landmark, "visibility", 1.0)),
                )

        return Pose2DFrame(
            camera_id=camera_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            landmarks=landmarks,
        )


def create_pose_estimator(name: str, min_confidence: float) -> BasePoseEstimator:
    normalized = name.lower().strip()
    if normalized == "mediapipe":
        return MediaPipePoseEstimator(min_confidence=min_confidence)
    raise ValueError(f"Unsupported pose backend: {name}")
