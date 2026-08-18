from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_counting import (  # noqa: E402
    STANDARD_COLUMNS,
    crossing_counts,
    crossing_error_metrics,
    frame_count_metrics,
)


def tracks(rows):
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


class EvaluateCountingTests(unittest.TestCase):
    def test_frame_count_metrics_measure_vehicle_count_error(self) -> None:
        ground_truth = tracks(
            [
                (1, 1, "car", 1.0, 0, 0, 10, 10),
                (2, 1, "car", 1.0, 0, 0, 10, 10),
            ]
        )
        predictions = tracks(
            [
                (1, 5, "car", 0.9, 0, 0, 10, 10),
                (1, 6, "car", 0.8, 0, 0, 10, 10),
            ]
        )
        result = frame_count_metrics(
            ground_truth, predictions, frame_indices=[1, 2], class_names=["car"]
        )
        self.assertEqual(result["mae_vehicles_per_frame"], 1.0)
        self.assertEqual(result["wape"], 1.0)

    def test_crossing_counts_use_production_hysteresis_logic(self) -> None:
        trajectory = tracks(
            [
                (1, 1, "car", 1.0, 40, 20, 20, 20),
                (2, 1, "car", 1.0, 40, 70, 20, 20),
            ]
        )
        result = crossing_counts(
            trajectory,
            frame_indices=[1, 2],
            frame_width=100,
            frame_height=100,
            class_to_id={"car": 4},
            line_y=0.5,
            line_tolerance_px=1.0,
            occupancy_grid_size_px=8,
        )
        self.assertEqual(result["down"]["car"], 1)

    def test_crossing_error_penalizes_false_and_missed_counts(self) -> None:
        result = crossing_error_metrics(
            [
                {"ground_truth_count": 3, "predicted_count": 1},
                {"ground_truth_count": 0, "predicted_count": 2},
            ]
        )
        self.assertEqual(result["absolute_error"], 4)
        self.assertAlmostEqual(result["wape"], 4 / 3)


if __name__ == "__main__":
    unittest.main()
