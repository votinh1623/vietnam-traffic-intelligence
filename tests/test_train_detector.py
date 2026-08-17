from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

from train_detector import load_config, training_arguments  # noqa: E402


class TrainDetectorTests(unittest.TestCase):
    def test_primary_config_never_selects_test(self) -> None:
        path = PROJECT_ROOT / "configs" / "experiments" / "yolov8s_v4_seed0.yaml"
        config = load_config(path)
        self.assertEqual(config["dataset"]["selection_split"], "validation")
        self.assertEqual(config["dataset"]["forbidden_selection_split"], "test")
        arguments = training_arguments(config, smoke=False)
        self.assertNotIn("test", arguments)
        self.assertEqual(arguments["freeze"], 0)

    def test_smoke_overrides_are_small(self) -> None:
        path = PROJECT_ROOT / "configs" / "experiments" / "yolov8s_v4_seed0.yaml"
        config = load_config(path)
        arguments = training_arguments(config, smoke=True)
        self.assertEqual(arguments["epochs"], 1)
        self.assertEqual(arguments["imgsz"], 640)
        self.assertEqual(arguments["fraction"], 0.10)


if __name__ == "__main__":
    unittest.main()

