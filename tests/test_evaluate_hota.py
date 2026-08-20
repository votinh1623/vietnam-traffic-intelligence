from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from evaluate_hota import evaluate_sequence  # noqa: E402

    TRACKEVAL_AVAILABLE = True
except ImportError:
    TRACKEVAL_AVAILABLE = False


def records(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["frame", "id", "class_name", "x", "y", "w", "h"]
    )


@unittest.skipUnless(TRACKEVAL_AVAILABLE, "trackeval is not installed")
class EvaluateSequenceTests(unittest.TestCase):
    def test_gt_only_frame_is_not_dropped_from_a_sequence(self) -> None:
        # Direct regression test for the fixed bug at the actual CLI
        # boundary (evaluate_hota.py's own frame handling), not just
        # hota_metrics.compute_hota, which was already correct before that
        # bug existed here -- so testing only compute_hota would not have
        # caught, and would not catch a future regression of, a re-added
        # pre-filter in this file. Frame 2 has real GT ("car") but the
        # tracker produced zero predictions for the whole frame -- exactly
        # the severe-occlusion failure mode this project cares about.
        ground_truth = records(
            (1, 1, "car", 0, 0, 10, 10),
            (2, 1, "car", 0, 0, 10, 10),
            (3, 1, "car", 0, 0, 10, 10),
        )
        predictions = records(
            (1, 1, "car", 0, 0, 10, 10),
            (3, 1, "car", 0, 0, 10, 10),
        )
        results = evaluate_sequence("seq", ground_truth, predictions, ["car"])
        self.assertEqual(set(results), {"seq:car"})
        result = results["seq:car"]
        # A regression that pre-filters GT to predicted frames would drop
        # frame 2 entirely and score this identically to a perfect track
        # (HOTA/DetA == 1.0). The correct evaluator must show a real,
        # non-perfect detection penalty from that missed frame.
        self.assertLess(float(np.mean(result["DetA"])), 1.0)
        self.assertLess(float(np.mean(result["HOTA"])), 1.0)

    def test_class_with_no_ground_truth_or_predictions_is_still_keyed(self) -> None:
        ground_truth = records((1, 1, "car", 0, 0, 10, 10))
        predictions = records((1, 1, "car", 0, 0, 10, 10))
        results = evaluate_sequence(
            "seq", ground_truth, predictions, ["car", "truck"]
        )
        self.assertEqual(set(results), {"seq:car", "seq:truck"})


if __name__ == "__main__":
    unittest.main()
