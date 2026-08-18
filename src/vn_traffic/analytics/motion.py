"""Frame-to-frame global motion compensation (GMC) for a panning/zooming camera.

This estimates only the 2D image-plane motion induced by the camera (modeled
as a similarity/Euclidean warp between consecutive frames) and accumulates it
against the first frame seen. It is not 3D pose, GPS, or BEV georeferencing,
and it does not correct for parallax between objects at different depths. Its
only job is to answer "where, in the current frame, is the region I marked on
frame 0" well enough to keep an ROI/counting-line roughly ground-anchored
while the camera pans or zooms, which a static polygon cannot do (see
experiments/uav_pipeline_e2e_v1_20260818 for the failure this replaces).

ECC alignment has a local convergence basin: it is reliable for the small
frame-to-frame motion typical of continuous video (a few pixels at a
moderate downscale), not for an arbitrarily large single-step jump or a hard
scene cut. When estimation fails to converge, `update()` freezes the last
known cumulative transform and reports the failure via `consecutive_failures`
rather than silently resetting to identity or guessing.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


PixelPoint = tuple[float, float]

IDENTITY_TRANSFORM = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Return the 2x3 affine equivalent to applying `inner` then `outer`."""
    outer_h = np.vstack([outer, [0.0, 0.0, 1.0]])
    inner_h = np.vstack([inner, [0.0, 0.0, 1.0]])
    composed_h = outer_h @ inner_h
    return composed_h[:2, :]


class GlobalMotionCompensator:
    """Track the affine transform from the reference (first) frame to the
    current frame using ECC image alignment, so a polygon drawn once on the
    reference frame can be re-projected into every later frame's pixels.
    """

    def __init__(
        self,
        *,
        downscale: int = 4,
        max_iterations: int = 200,
        termination_eps: float = 1e-6,
    ):
        if downscale < 1:
            raise ValueError("downscale must be at least one")
        self.downscale = downscale
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iterations,
            termination_eps,
        )
        self._previous_small_gray: np.ndarray | None = None
        self.cumulative_transform = IDENTITY_TRANSFORM.copy()
        self.consecutive_failures = 0
        self.frames_seen = 0

    def _to_small_gray(self, frame_bgr: Any) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale == 1:
            return gray.astype(np.float32)
        height, width = gray.shape
        small = cv2.resize(
            gray,
            (max(1, width // self.downscale), max(1, height // self.downscale)),
            interpolation=cv2.INTER_AREA,
        )
        return small.astype(np.float32)

    def update(self, frame_bgr: Any) -> bool:
        """Feed the next frame in sequence. Returns whether this frame's
        motion was estimated successfully. On failure the last valid
        cumulative transform is held (frozen), not reset to identity, and
        `consecutive_failures` increments so callers can flag degraded
        localization confidence rather than silently trusting a stale warp.
        """
        small_gray = self._to_small_gray(frame_bgr)
        self.frames_seen += 1
        if self._previous_small_gray is None:
            self._previous_small_gray = small_gray
            return True

        warp_guess = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_guess = cv2.findTransformECC(
                self._previous_small_gray,
                small_gray,
                warp_guess,
                cv2.MOTION_EUCLIDEAN,
                self._criteria,
            )
            # Empirically verified in test_motion.py against a known
            # synthetic shift (not derived from the OpenCV docs' wording
            # alone, which is easy to misread here): the raw warp returned by
            # findTransformECC(template=previous, input=current, ...) already
            # maps previous-frame pixels to current-frame pixels once the
            # downscale factor is corrected, so no extra inversion is applied.
            previous_to_current = np.array(warp_guess, dtype=np.float64)
            previous_to_current[:, 2] *= self.downscale
            success = True
        except cv2.error:
            previous_to_current = None
            success = False

        self._previous_small_gray = small_gray
        if success and previous_to_current is not None:
            self.consecutive_failures = 0
            self.cumulative_transform = _compose_affine(
                previous_to_current, self.cumulative_transform
            )
        else:
            self.consecutive_failures += 1
        return success

    def warp_points(
        self, points_xy: Sequence[PixelPoint]
    ) -> list[PixelPoint]:
        """Map reference-frame (frame 0) pixel points into current-frame pixels."""
        matrix = self.cumulative_transform
        result = []
        for x, y in points_xy:
            warped_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
            warped_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
            result.append((float(warped_x), float(warped_y)))
        return result
