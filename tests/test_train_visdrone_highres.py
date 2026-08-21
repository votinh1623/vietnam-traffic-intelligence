from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

from train_visdrone_highres import load_config, training_arguments  # noqa: E402


class VisDroneHighResolutionTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = (
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "yolov8s_visdrone_highres_ft_v1.yaml"
        )
        self.config = load_config(self.path)

    def test_protocol_targets_validation_and_forbids_test(self) -> None:
        self.assertEqual(self.config["dataset"]["selection_split"], "val")
        self.assertEqual(self.config["dataset"]["forbidden_split"], "test")

    def test_pilot_is_direct_high_resolution_fine_tuning(self) -> None:
        arguments = training_arguments(self.config, smoke=False)
        self.assertEqual(arguments["imgsz"], 1280)
        self.assertEqual(arguments["batch"], 2)
        self.assertEqual(arguments["mosaic"], 0.0)
        self.assertEqual(arguments["mixup"], 0.0)
        self.assertEqual(arguments["freeze"], 0)
        self.assertEqual(arguments["epochs"], 5)

    def test_smoke_keeps_resolution_and_memory_sensitive_batch(self) -> None:
        arguments = training_arguments(self.config, smoke=True)
        self.assertEqual(arguments["epochs"], 1)
        self.assertEqual(arguments["imgsz"], 1280)
        self.assertEqual(arguments["batch"], 2)
        self.assertEqual(arguments["fraction"], 0.02)

    def test_gates_are_frozen_against_existing_coco_baseline(self) -> None:
        gates = self.config["gates"]
        self.assertAlmostEqual(gates["baseline_coco_ap_small"], 0.19380973711353866)
        self.assertEqual(gates["ap_small_absolute_gain_min"], 0.01)
        self.assertEqual(gates["overall_ap_absolute_drop_max"], 0.005)


if __name__ == "__main__":
    unittest.main()
