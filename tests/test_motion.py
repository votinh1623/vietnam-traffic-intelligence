from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.analytics.motion import GlobalMotionCompensator  # noqa: E402


def _synthetic_textured_frame(
    width: int = 160, height: int = 120, seed: int = 0
) -> np.ndarray:
    # sigma=2.0 gives ECC a smooth-enough gradient landscape to converge to
    # the true optimum instead of a nearby local minimum; this was verified
    # empirically across several seeds, not assumed.
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def _shift_frame(frame: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    height, width = frame.shape[:2]
    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT101
    )


class GlobalMotionCompensatorTests(unittest.TestCase):
    def test_recovers_a_known_pure_translation(self) -> None:
        frame0 = _synthetic_textured_frame(seed=0)
        shift_x, shift_y = 8.0, -5.0
        frame1 = _shift_frame(frame0, shift_x, shift_y)

        gmc = GlobalMotionCompensator(downscale=1)
        self.assertTrue(gmc.update(frame0))
        self.assertTrue(gmc.update(frame1))

        # Content at (x0, y0) on frame0 was moved to (x0+shift_x, y0+shift_y)
        # on frame1 by construction, so warping a reference-frame point should
        # recover that same displacement. This is the check that would catch
        # a reversed transform direction (a wrong sign would fail loudly).
        (warped_x, warped_y), = gmc.warp_points([(80.0, 60.0)])
        self.assertAlmostEqual(warped_x, 80.0 + shift_x, delta=1.5)
        self.assertAlmostEqual(warped_y, 60.0 + shift_y, delta=1.5)

    def test_accumulates_across_multiple_frames(self) -> None:
        frame0 = _synthetic_textured_frame(seed=1)
        per_step = (3.0, 2.0)
        frame1 = _shift_frame(frame0, *per_step)
        frame2 = _shift_frame(frame0, per_step[0] * 2, per_step[1] * 2)

        gmc = GlobalMotionCompensator(downscale=1)
        gmc.update(frame0)
        gmc.update(frame1)
        gmc.update(frame2)

        (warped_x, warped_y), = gmc.warp_points([(50.0, 40.0)])
        self.assertAlmostEqual(warped_x, 50.0 + per_step[0] * 2, delta=2.0)
        self.assertAlmostEqual(warped_y, 40.0 + per_step[1] * 2, delta=2.0)

    def test_identity_before_any_motion(self) -> None:
        gmc = GlobalMotionCompensator(downscale=2)
        gmc.update(_synthetic_textured_frame(seed=2))
        points = gmc.warp_points([(10.0, 20.0), (30.0, 40.0)])
        self.assertEqual(points, [(10.0, 20.0), (30.0, 40.0)])

    def test_downscale_still_recovers_translation_in_full_resolution_pixels(
        self,
    ) -> None:
        # 30 fps real video moves only a few pixels between consecutive
        # frames, not tens of pixels; ECC's convergence basin is local, so
        # this uses a realistic inter-frame shift rather than an extreme one,
        # and a frame large enough that a 4x downscale still leaves usable
        # texture (real UAV frames are far bigger than this).
        frame0 = _synthetic_textured_frame(width=640, height=480, seed=3)
        shift_x, shift_y = 6.0, -3.0
        frame1 = _shift_frame(frame0, shift_x, shift_y)

        gmc = GlobalMotionCompensator(downscale=4)
        gmc.update(frame0)
        gmc.update(frame1)

        (warped_x, warped_y), = gmc.warp_points([(320.0, 240.0)])
        self.assertAlmostEqual(warped_x, 320.0 + shift_x, delta=3.0)
        self.assertAlmostEqual(warped_y, 240.0 + shift_y, delta=3.0)

    def test_failed_estimate_freezes_last_transform_and_counts_failure(self) -> None:
        frame0 = _synthetic_textured_frame(seed=0)
        shift_x, shift_y = 6.0, 3.0
        frame1 = _shift_frame(frame0, shift_x, shift_y)
        blank = np.zeros_like(frame0)

        gmc = GlobalMotionCompensator(downscale=1)
        gmc.update(frame0)
        gmc.update(frame1)
        transform_after_good_estimate = gmc.cumulative_transform.copy()

        gmc.update(blank)
        gmc.update(blank)

        self.assertGreaterEqual(gmc.consecutive_failures, 1)
        np.testing.assert_allclose(
            gmc.cumulative_transform, transform_after_good_estimate
        )

    def test_total_failures_does_not_reset_on_recovery(self) -> None:
        # consecutive_failures alone can read 0 at the end of a run that
        # genuinely lost lock earlier and recovered -- total_failures is
        # the run-wide count a caller needs to know that happened at all.
        frame0 = _synthetic_textured_frame(seed=4)
        shift_x, shift_y = 5.0, 2.0
        frame1 = _shift_frame(frame0, shift_x, shift_y)
        blank = np.zeros_like(frame0)

        gmc = GlobalMotionCompensator(downscale=1)
        gmc.update(frame0)
        gmc.update(frame1)  # good pair: establishes real motion history
        gmc.update(blank)  # fails (frame1 -> blank)
        gmc.update(blank)  # fails (blank -> blank)
        failures_after_blanks = gmc.total_failures
        self.assertGreaterEqual(failures_after_blanks, 2)

        gmc.update(frame0)  # still fails (blank -> frame0)
        gmc.update(frame1)  # recovers: (frame0 -> frame1) is a real pair again

        self.assertEqual(gmc.consecutive_failures, 0)
        self.assertGreaterEqual(gmc.total_failures, failures_after_blanks + 1)


if __name__ == "__main__":
    unittest.main()
