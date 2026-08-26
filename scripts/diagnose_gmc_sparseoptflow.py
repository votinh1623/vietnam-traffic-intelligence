"""Replicate BoT-SORT's internal sparseOptFlow GMC frame-by-frame to check
for silent identity-transform fallback (Ultralytics logs a warning but this
project's pipeline never surfaces it) and measure how much real camera
motion/zoom it is actually detecting and correcting.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np


FEATURE_PARAMS = {
    "maxCorners": 1000,
    "qualityLevel": 0.01,
    "minDistance": 1,
    "blockSize": 3,
    "useHarrisDetector": False,
    "k": 0.04,
}
DOWNSCALE = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    prev_frame = None
    prev_keypoints = None
    initialized = False

    total = 0
    fallback_identity = 0
    scales = []
    translations = []
    keypoint_counts = []
    matched_counts = []
    fallback_frames: list[int] = []

    frame_index = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        height, width = raw.shape[:2]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if DOWNSCALE > 1:
            gray = cv2.resize(gray, (width // DOWNSCALE, height // DOWNSCALE))

        keypoints = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
        keypoint_counts.append(0 if keypoints is None else len(keypoints))

        if not initialized or prev_keypoints is None:
            prev_frame = gray.copy()
            prev_keypoints = copy.copy(keypoints)
            initialized = True
            frame_index += 1
            continue

        total += 1
        matched, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_keypoints, None)
        good = status.ravel().astype(bool)
        prev_points = prev_keypoints[good]
        curr_points = matched[good]
        matched_counts.append(len(prev_points))

        if prev_points.shape[0] > 4:
            transform, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, cv2.RANSAC)
            if transform is None:
                fallback_identity += 1
                fallback_frames.append(frame_index)
            else:
                a, b = transform[0, 0], transform[0, 1]
                scale = float(np.hypot(a, b))
                tx, ty = transform[0, 2] * DOWNSCALE, transform[1, 2] * DOWNSCALE
                scales.append(scale)
                translations.append(float(np.hypot(tx, ty)))
        else:
            fallback_identity += 1
            fallback_frames.append(frame_index)

        prev_frame = gray.copy()
        prev_keypoints = copy.copy(keypoints)
        frame_index += 1

    print(f"source: {args.source}")
    print(f"frames compared: {total}")
    print(f"identity fallback (not enough matched points, <=4): {fallback_identity} ({fallback_identity/total:.1%})")
    print(f"keypoints found per frame: min={min(keypoint_counts)} median={sorted(keypoint_counts)[len(keypoint_counts)//2]} max={max(keypoint_counts)}")
    if matched_counts:
        print(f"matched points per compared frame: min={min(matched_counts)} median={sorted(matched_counts)[len(matched_counts)//2]} max={max(matched_counts)}")
    if scales:
        print(f"estimated scale factor per frame: min={min(scales):.4f} median={sorted(scales)[len(scales)//2]:.4f} max={max(scales):.4f}")
    if translations:
        print(f"estimated translation magnitude (px, original res): min={min(translations):.1f} median={sorted(translations)[len(translations)//2]:.1f} max={max(translations):.1f}")
    if fallback_frames:
        print(f"first 20 fallback frame indices: {fallback_frames[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
