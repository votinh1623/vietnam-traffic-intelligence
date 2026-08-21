from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_proposal_oracle import (  # noqa: E402
    add_temporal_support,
    classify_observations,
    maximum_cardinality_matches,
    summarize,
)
from export_lowconf_proposals import export_sequence  # noqa: E402


GT_COLUMNS = [
    "sequence", "frame", "id", "class_name", "x", "y", "w", "h",
    "truncation", "occlusion",
]
PRED_COLUMNS = [
    "sequence", "frame", "id", "class_name", "confidence", "x", "y", "w", "h",
]
PROPOSAL_COLUMNS = [
    "sequence", "frame", "proposal_id", "class_name", "confidence", "x", "y", "w", "h",
]


class TvlrOracleTests(unittest.TestCase):
    def test_maximum_matching_does_not_reuse_one_proposal(self) -> None:
        matrix = np.array([[0.9], [0.8]])
        matches = maximum_cardinality_matches(matrix, 0.5)
        self.assertEqual(len(matches), 1)

    def test_classification_counts_only_increment_beyond_bytetrack(self) -> None:
        ground_truth = pd.DataFrame(
            [
                ("seq", 1, 10, "car", 0, 0, 10, 10, 0, 0),
                ("seq", 1, 11, "car", 20, 0, 10, 10, 0, 1),
                ("seq", 1, 12, "car", 40, 0, 10, 10, 0, 2),
            ],
            columns=GT_COLUMNS,
        )
        baseline = pd.DataFrame(
            [("seq", 1, 1, "car", 0.8, 0, 0, 10, 10)], columns=PRED_COLUMNS
        )
        proposals = pd.DataFrame(
            [
                ("seq", 1, 0, "car", 0.08, 20, 0, 10, 10),
                ("seq", 1, 1, "truck", 0.9, 40, 0, 10, 10),
            ],
            columns=PROPOSAL_COLUMNS,
        )
        result = classify_observations(
            ground_truth,
            baseline,
            proposals,
            iou_threshold=0.5,
            track_low_thresh=0.1,
            track_high_thresh=0.5,
        ).sort_values("id")
        self.assertEqual(
            result["status"].tolist(),
            ["bytetrack_caught", "incremental_candidate", "no_post_nms_proposal"],
        )
        self.assertEqual(result.iloc[1]["proposal_score_band"], "below_track_low")

    def test_temporal_support_separates_one_sided_and_bracketed(self) -> None:
        observations = pd.DataFrame(
            [
                ("seq", 1, 10, "bytetrack_caught"),
                ("seq", 2, 10, "incremental_candidate"),
                ("seq", 3, 10, "bytetrack_caught"),
                ("seq", 1, 11, "incremental_candidate"),
                ("seq", 2, 11, "bytetrack_caught"),
            ],
            columns=["sequence", "frame", "id", "status"],
        )
        result = add_temporal_support(observations, [1])
        bracketed = result[(result["id"] == 10) & (result["frame"] == 2)].iloc[0]
        one_sided = result[(result["id"] == 11) & (result["frame"] == 1)].iloc[0]
        self.assertTrue(bracketed["support_forward_backward_1"])
        self.assertTrue(one_sided["support_one_sided_1"])
        self.assertFalse(one_sided["support_forward_backward_1"])

    def test_summary_applies_locked_incremental_gates(self) -> None:
        observations = pd.DataFrame(
            [
                ("seq", 1, 1, "car", "bytetrack_caught", 10.0, 0, False),
                ("seq", 1, 2, "car", "incremental_candidate", 10.0, 1, True),
                ("seq", 2, 2, "car", "incremental_candidate", 10.0, 1, True),
            ],
            columns=[
                "sequence", "frame", "id", "class_name", "status", "sqrt_area_px", "occlusion",
                "support_one_sided_1",
            ],
        )
        observations["support_forward_backward_1"] = observations["support_one_sided_1"]
        observations["proposal_score_band"] = [None, "below_track_low", "below_track_low"]
        baseline = pd.DataFrame(
            [
                ("seq", 1, 1, "car", 0.8, 0, 0, 10, 10),
                ("seq", 1, 99, "person", 0.9, 0, 0, 10, 10),
            ],
            columns=PRED_COLUMNS,
        )
        result = summarize(
            observations,
            baseline,
            windows=[1],
            tiny_max_sqrt_area_px=16,
            occluded_min_level=1,
            gates={
                "incremental_candidate_fraction_tiny_or_occluded_min": 0.15,
                "oracle_detection_recall_absolute_gain_min": 0.02,
                "oracle_frame_count_wape_relative_reduction_min": 0.05,
                "require_nonzero_temporal_support": True,
            },
        )
        self.assertTrue(result["proceed_to_stage_c"])
        self.assertEqual(result["incremental_candidates"], 2)
        self.assertEqual(result["oracle_frame_count_wape"], 0.0)

    def test_export_reports_max_det_saturation_and_filters_classes(self) -> None:
        class FakeBoxes:
            def __init__(self) -> None:
                self.cls = torch.tensor([0, 1])
                self.conf = torch.tensor([0.08, 0.9])
                self.xyxy = torch.tensor([[1, 2, 11, 12], [0, 0, 4, 4]])

            def __len__(self) -> int:
                return len(self.cls)

        class FakeModel:
            names = {0: "car", 1: "person"}

            def predict(self, _image, **_kwargs):
                return [SimpleNamespace(boxes=FakeBoxes())]

        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "0000001.jpg"
            cv2.imwrite(str(frame), np.zeros((16, 16, 3), dtype=np.uint8))
            proposals, saturated = export_sequence(
                FakeModel(),
                sequence="seq",
                frame_paths=[frame],
                settings={"imgsz": 1280, "confidence": 0.02, "iou": 0.7, "max_det": 2, "device": "0"},
                allowed_classes={"car"},
            )
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals.iloc[0]["class_name"], "car")
            self.assertEqual(saturated, [1])


if __name__ == "__main__":
    unittest.main()
