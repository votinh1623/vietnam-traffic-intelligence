from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tracking_metrics import (  # noqa: E402
    compute_motchallenge_metrics,
    iou_distance_matrix,
)


def records(*rows):
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])


class TrackingMetricTests(unittest.TestCase):
    def test_identical_boxes_have_zero_distance(self) -> None:
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        distances = iou_distance_matrix(boxes, boxes, iou_threshold=0.5)
        self.assertEqual(distances.shape, (1, 1))
        self.assertAlmostEqual(distances[0, 0], 0.0)

    def test_non_overlapping_boxes_are_gated(self) -> None:
        ground_truth = np.array([[0.0, 0.0, 10.0, 10.0]])
        prediction = np.array([[20.0, 20.0, 10.0, 10.0]])
        distances = iou_distance_matrix(
            ground_truth, prediction, iou_threshold=0.5
        )
        self.assertTrue(np.isnan(distances[0, 0]))

    def test_perfect_track_scores_one(self) -> None:
        ground_truth = records((1, 1, 0, 0, 10, 10), (2, 1, 1, 0, 10, 10))
        prediction = ground_truth.copy()
        summary = compute_motchallenge_metrics(ground_truth, prediction)
        self.assertAlmostEqual(summary.loc["acc", "mota"], 1.0)
        self.assertAlmostEqual(summary.loc["acc", "idf1"], 1.0)
        self.assertAlmostEqual(summary.loc["acc", "motp"], 0.0)

    def test_prediction_only_frame_counts_as_false_positive(self) -> None:
        ground_truth = records((1, 1, 0, 0, 10, 10))
        prediction = records((1, 1, 0, 0, 10, 10), (2, 2, 0, 0, 10, 10))
        summary = compute_motchallenge_metrics(ground_truth, prediction)
        self.assertEqual(summary.loc["acc", "num_false_positives"], 1)
        self.assertAlmostEqual(summary.loc["acc", "mota"], 0.0)

    def test_iou_threshold_controls_matching(self) -> None:
        ground_truth = records((1, 1, 0, 0, 10, 10))
        prediction = records((1, 1, 4, 0, 10, 10))  # IoU = 6/14 ~= 0.429
        loose = compute_motchallenge_metrics(
            ground_truth, prediction, iou_threshold=0.4
        )
        strict = compute_motchallenge_metrics(
            ground_truth, prediction, iou_threshold=0.5
        )
        self.assertAlmostEqual(loose.loc["acc", "mota"], 1.0)
        self.assertEqual(loose.loc["acc", "num_misses"], 0)
        self.assertAlmostEqual(strict.loc["acc", "mota"], -1.0)
        self.assertEqual(strict.loc["acc", "num_misses"], 1)


if __name__ == "__main__":
    unittest.main()
