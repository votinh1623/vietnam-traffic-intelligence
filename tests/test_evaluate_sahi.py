from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_sahi import (  # noqa: E402
    evaluate_coco,
    parse_visdrone_annotation,
    validate_config,
)


class EvaluateSahiTests(unittest.TestCase):
    def test_parser_excludes_ignored_and_clamps_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.txt"
            path.write_text(
                "-5,-2,20,12,1,4,0,0\n"
                "0,0,10,10,0,4,0,0\n"
                "0,0,10,10,1,0,0,0\n",
                encoding="utf-8",
            )
            annotations = parse_visdrone_annotation(
                path, width=100, height=80, class_count=10
            )
            self.assertEqual(len(annotations), 1)
            self.assertEqual(annotations[0]["category_id"], 4)
            self.assertEqual(annotations[0]["bbox"], [0.0, 0.0, 15.0, 10.0])

    def test_perfect_coco_prediction_scores_one(self) -> None:
        ground_truth = {
            "info": {},
            "licenses": [],
            "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
        }
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}
        ]
        metrics = evaluate_coco(ground_truth, predictions, [1, 100, 1000])
        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertAlmostEqual(metrics["ap50"], 1.0)
        self.assertEqual(metrics["max_dets"], 1000)

    def test_coco_ap_uses_last_max_dets_setting(self) -> None:
        annotations = []
        predictions = []
        for index in range(101):
            x = (index % 11) * 10
            y = (index // 11) * 10
            bbox = [x, y, 5, 5]
            annotations.append(
                {
                    "id": index + 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": 25,
                    "iscrowd": 0,
                }
            )
            predictions.append(
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": bbox,
                    "score": 1.0 - index * 0.001,
                }
            )
        ground_truth = {
            "info": {},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "dense.jpg", "width": 120, "height": 100}
            ],
            "categories": [{"id": 1, "name": "vehicle"}],
            "annotations": annotations,
        }

        metrics = evaluate_coco(ground_truth, predictions, [1, 100, 1000])

        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertAlmostEqual(metrics["ar_max_dets"], 1.0)

    def test_selection_benchmark_refuses_test_split(self) -> None:
        config = {
            "dataset": {"split": "test"},
            "modes": {"standard": {"type": "standard"}},
        }
        with self.assertRaisesRegex(ValueError, "split: validation"):
            validate_config(config, "standard")

    def test_sahi_mode_requires_explicit_postprocess_policy(self) -> None:
        config = {
            "dataset": {"split": "validation"},
            "model": {"confidence": 0.01},
            "modes": {"sahi": {"type": "sahi"}},
        }
        with self.assertRaisesRegex(KeyError, "force_postprocess_type"):
            validate_config(config, "sahi")


if __name__ == "__main__":
    unittest.main()
