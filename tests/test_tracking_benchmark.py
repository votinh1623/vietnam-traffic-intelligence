from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_tracking import (  # noqa: E402
    TRACK_COLUMNS,
    class_accumulators,
    file_manifest_sha256,
    load_visdrone_ground_truth,
    sequence_frames,
    track_sequence,
)
from tracking_metrics import compute_many_motchallenge_metrics  # noqa: E402


class TrackingBenchmarkTests(unittest.TestCase):
    def test_loads_only_valid_mapped_visdrone_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "sequence.txt"
            annotation.write_text(
                "1,10,0,0,20,10,1,4,0,0\n"
                "1,11,0,0,20,10,1,5,0,0\n"
                "1,12,0,0,20,10,0,4,0,0\n"
                "1,13,0,0,0,10,1,4,0,0\n",
                encoding="utf-8",
            )
            result = load_visdrone_ground_truth(annotation, {4: "car"})
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["class_name"], "car")
            self.assertEqual(result.iloc[0]["id"], 10)

    def test_sequence_frames_are_numeric_and_limitable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("0000010.jpg", "0000002.jpg", "0000001.jpg"):
                (root / name).touch()
            result = sequence_frames(root, max_frames=2)
            self.assertEqual([path.stem for path in result], ["0000001", "0000002"])

    def test_file_manifest_hash_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "0000001.jpg"
            second = root / "0000002.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            before = file_manifest_sha256([first, second], root)
            second.write_bytes(b"changed")
            after = file_manifest_sha256([first, second], root)
            self.assertNotEqual(before, after)

    def test_class_mismatch_is_not_matched_in_overall_metrics(self) -> None:
        ground_truth = pd.DataFrame(
            [(1, 1, "car", 0, 0, 10, 10)],
            columns=["frame", "id", "class_name", "x", "y", "w", "h"],
        )
        predictions = pd.DataFrame(
            [("seq", 1, 5, "truck", 0.9, 0, 0, 10, 10)],
            columns=TRACK_COLUMNS,
        )
        accumulators, names = class_accumulators(
            ground_truth,
            predictions,
            sequence="seq",
            class_names=["car", "truck"],
            iou_threshold=0.5,
        )
        summary = compute_many_motchallenge_metrics(accumulators, names)
        self.assertEqual(summary.loc["OVERALL", "num_misses"], 1)
        self.assertEqual(summary.loc["OVERALL", "num_false_positives"], 1)
        self.assertEqual(summary.loc["OVERALL", "num_matches"], 0)
        self.assertEqual(summary.loc["OVERALL", "idf1"], 0)

    def test_tracking_forwards_dense_scene_max_det(self) -> None:
        class FakeModel:
            predictor = None
            names = {0: "car"}

            def __init__(self) -> None:
                self.kwargs = None

            def track(self, _image, **kwargs):
                self.kwargs = kwargs
                return [SimpleNamespace(boxes=None)]

        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "0000001.jpg"
            cv2.imwrite(str(frame), np.zeros((16, 16, 3), dtype=np.uint8))
            model = FakeModel()
            result = track_sequence(
                model,
                sequence="dense",
                frame_paths=[frame],
                perception={
                    "tracker": "bytetrack_custom.yaml",
                    "imgsz": 1280,
                    "confidence": 0.1,
                    "iou": 0.7,
                    "max_det": 1000,
                    "device": "0",
                },
            )
            self.assertTrue(result.empty)
            self.assertEqual(model.kwargs["max_det"], 1000)


if __name__ == "__main__":
    unittest.main()
