from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for feature matching. Install it with `pip install opencv-contrib-python`."
        )


def detect_and_match_features(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    top_k: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    _require_cv2()

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        extractor = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        extractor = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    keypoints_a, desc_a = extractor.detectAndCompute(gray_a, None)
    keypoints_b, desc_b = extractor.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        return np.empty((0, 2)), np.empty((0, 2))

    raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good.append(best)

    good = sorted(good, key=lambda item: item.distance)[:top_k]
    pts_a = np.asarray([keypoints_a[m.queryIdx].pt for m in good], dtype=float)
    pts_b = np.asarray([keypoints_b[m.trainIdx].pt for m in good], dtype=float)
    return pts_a, pts_b


def estimate_fundamental_matrix(frame_a: np.ndarray, frame_b: np.ndarray) -> Optional[np.ndarray]:
    pts_a, pts_b = detect_and_match_features(frame_a, frame_b)
    if len(pts_a) < 8:
        return None
    fundamental, _ = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.5, 0.99)
    if fundamental is None or fundamental.shape != (3, 3):
        return None
    norm = np.linalg.norm(fundamental)
    if norm == 0:
        return None
    return fundamental / norm
