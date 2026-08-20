from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.analytics.stillness import (  # noqa: E402
    StillnessHeatmapRenderer,
    grid_mean,
    render_heatmap_overlay,
    stalled_dense_fraction,
    stalled_dense_mask,
    stalled_dense_score,
    texture_density,
)


def _textured_frame(width: int = 160, height: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), sigmaX=1.0)


def _flat_frame(width: int = 160, height: int = 120, value: int = 128) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _shift_frame(frame: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    height, width = frame.shape[:2]
    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT101
    )


class StillnessTests(unittest.TestCase):
    def test_static_packed_texture_is_flagged_stalled_dense(self) -> None:
        # Same textured frame twice: no motion, high local detail -- the
        # target case (a stopped, packed crowd the detector cannot resolve).
        frame = _textured_frame(seed=0)
        mask = stalled_dense_mask(
            frame, frame, cell_px=8, motion_threshold=0.5, texture_threshold=5.0
        )
        self.assertGreater(stalled_dense_fraction(mask), 0.8)

    def test_static_empty_road_is_not_flagged(self) -> None:
        # No motion, but also no texture (plain asphalt) -- must not be
        # confused with a packed cluster.
        frame = _flat_frame()
        mask = stalled_dense_mask(
            frame, frame, cell_px=8, motion_threshold=0.5, texture_threshold=5.0
        )
        self.assertEqual(stalled_dense_fraction(mask), 0.0)

    def test_moving_textured_region_is_not_flagged(self) -> None:
        # High texture, but genuinely moving -- ordinary flowing traffic,
        # not a stall.
        frame0 = _textured_frame(seed=1)
        frame1 = _shift_frame(frame0, 6.0, 0.0)
        mask = stalled_dense_mask(
            frame0, frame1, cell_px=8, motion_threshold=0.5, texture_threshold=5.0
        )
        self.assertLess(stalled_dense_fraction(mask), 0.2)

    def test_stalled_dense_fraction_respects_roi_mask(self) -> None:
        mask = np.array([[True, True], [False, False]])
        roi = np.array([[False, False], [True, True]])
        # The stalled cells are entirely outside the ROI, mirroring the
        # real run37 failure (the packed cluster sat outside the hand-drawn
        # ROI) -- restricting to the ROI must report zero, not the raw 0.5.
        self.assertEqual(stalled_dense_fraction(mask, roi_mask=roi), 0.0)

    def test_grid_mean_reduces_to_expected_shape(self) -> None:
        values = np.ones((80, 160), dtype=np.float32)
        reduced = grid_mean(values, cell_px=8)
        self.assertEqual(reduced.shape, (10, 20))
        np.testing.assert_allclose(reduced, 1.0)

    def test_texture_density_is_zero_on_a_flat_frame(self) -> None:
        frame = _flat_frame()
        self.assertTrue(np.allclose(texture_density(frame), 0.0))


def _half_textured_frame(width: int = 160, height: int = 120, seed: int = 0) -> np.ndarray:
    half = width // 2
    textured = _textured_frame(width=half, height=height, seed=seed)
    flat = _flat_frame(width=width - half, height=height, value=128)
    return np.concatenate([textured, flat], axis=1)


class StalledDenseScoreTests(unittest.TestCase):
    def test_static_textured_half_scores_high_flat_half_scores_zero(self) -> None:
        frame = _half_textured_frame(seed=0)
        score = stalled_dense_score(
            frame, frame, cell_px=8, motion_threshold=0.5, texture_percentile=90.0
        )
        grid_width = score.shape[1]
        left, right = score[:, : grid_width // 2], score[:, grid_width // 2 :]
        self.assertGreater(left.max(), 0.5)
        self.assertTrue(np.all(right == 0.0))

    def test_moving_textured_frame_scores_zero_everywhere(self) -> None:
        frame0 = _textured_frame(seed=2)
        frame1 = _shift_frame(frame0, 6.0, 0.0)
        score = stalled_dense_score(
            frame0, frame1, cell_px=8, motion_threshold=0.5, texture_percentile=90.0
        )
        self.assertTrue(np.all(score == 0.0))


class RenderHeatmapOverlayTests(unittest.TestCase):
    def test_zero_score_leaves_frame_unchanged(self) -> None:
        frame = np.full((64, 96, 3), 100, dtype=np.uint8)
        score = np.zeros((8, 12), dtype=np.float32)
        blended = render_heatmap_overlay(frame, score, alpha_max=0.5)
        np.testing.assert_array_equal(blended, frame)

    def test_hot_region_visibly_changes_the_frame(self) -> None:
        frame = np.full((64, 96, 3), 100, dtype=np.uint8)
        score = np.zeros((8, 12), dtype=np.float32)
        score[2:4, 2:4] = 1.0
        blended = render_heatmap_overlay(frame, score, alpha_max=0.5)
        self.assertFalse(np.array_equal(blended, frame))
        self.assertEqual(blended.shape, frame.shape)


class StillnessHeatmapRendererTests(unittest.TestCase):
    def test_first_call_returns_display_frame_unchanged(self) -> None:
        renderer = StillnessHeatmapRenderer(downscale=1, cell_px=8)
        raw = cv2.cvtColor(_half_textured_frame(seed=0), cv2.COLOR_GRAY2BGR)
        display = np.full_like(raw, 50)
        result = renderer.render(raw, display)
        np.testing.assert_array_equal(result, display)

    def test_second_call_tints_the_static_textured_region(self) -> None:
        renderer = StillnessHeatmapRenderer(
            downscale=1, cell_px=8, motion_threshold=0.5, texture_percentile=90.0
        )
        raw = cv2.cvtColor(_half_textured_frame(seed=1), cv2.COLOR_GRAY2BGR)
        display = raw.copy()
        renderer.render(raw, display)
        result = renderer.render(raw, display)
        self.assertFalse(np.array_equal(result, display))


if __name__ == "__main__":
    unittest.main()
