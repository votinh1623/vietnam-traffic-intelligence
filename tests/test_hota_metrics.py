from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from hota_metrics import (  # noqa: E402
        build_hota_data,
        compute_hota,
        compute_many_hota,
        iou_matrix,
    )

    TRACKEVAL_AVAILABLE = True
except ImportError:
    TRACKEVAL_AVAILABLE = False


def records(*rows):
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])


@unittest.skipUnless(TRACKEVAL_AVAILABLE, "trackeval is not installed")
class HotaMetricTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self) -> None:
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        ious = iou_matrix(boxes, boxes)
        self.assertEqual(ious.shape, (1, 1))
        self.assertAlmostEqual(ious[0, 0], 1.0)

    def test_non_overlapping_boxes_have_iou_zero(self) -> None:
        ground_truth = np.array([[0.0, 0.0, 10.0, 10.0]])
        prediction = np.array([[20.0, 20.0, 10.0, 10.0]])
        ious = iou_matrix(ground_truth, prediction)
        self.assertAlmostEqual(ious[0, 0], 0.0)

    def test_perfect_track_scores_hota_one(self) -> None:
        ground_truth = records((1, 1, 0, 0, 10, 10), (2, 1, 1, 0, 10, 10))
        prediction = ground_truth.copy()
        result = compute_hota(ground_truth, prediction)
        # HOTA/DetA/AssA are arrays over the 0.05..0.95 alpha sweep; a
        # perfect track scores 1.0 at every threshold.
        self.assertTrue(np.allclose(result["HOTA"], 1.0))
        self.assertTrue(np.allclose(result["DetA"], 1.0))
        self.assertTrue(np.allclose(result["AssA"], 1.0))

    def test_completely_missed_ground_truth_scores_hota_zero(self) -> None:
        ground_truth = records((1, 1, 0, 0, 10, 10))
        prediction = records()
        result = compute_hota(ground_truth, prediction)
        self.assertTrue(np.allclose(result["HOTA"], 0.0))

    def test_id_switch_reduces_association_but_not_detection(self) -> None:
        # Same boxes every frame (perfect detection), but the predicted
        # identity flips between frame 1 and frame 2 (one ID switch).
        ground_truth = records(
            (1, 1, 0, 0, 10, 10), (2, 1, 0, 0, 10, 10), (3, 1, 0, 0, 10, 10)
        )
        prediction = records(
            (1, 1, 0, 0, 10, 10), (2, 2, 0, 0, 10, 10), (3, 2, 0, 0, 10, 10)
        )
        result = compute_hota(ground_truth, prediction)
        self.assertTrue(np.allclose(result["DetA"], 1.0))
        self.assertLess(float(np.mean(result["AssA"])), 1.0)
        self.assertLess(float(np.mean(result["HOTA"])), 1.0)

    def test_combine_many_matches_single_sequence_result(self) -> None:
        ground_truth = records((1, 1, 0, 0, 10, 10), (2, 1, 1, 0, 10, 10))
        prediction = ground_truth.copy()
        data = build_hota_data(ground_truth, prediction)
        from trackeval.metrics import HOTA

        single = HOTA().eval_sequence(data)
        combined = compute_many_hota({"seq_a": single})
        self.assertAlmostEqual(combined["HOTA"], float(np.mean(single["HOTA"])))


if __name__ == "__main__":
    unittest.main()
