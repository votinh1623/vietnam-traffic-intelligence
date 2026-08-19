from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

from train_detector import load_config, training_arguments  # noqa: E402

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainDetectorTests(unittest.TestCase):
    def test_primary_config_never_selects_test(self) -> None:
        path = PROJECT_ROOT / "configs" / "experiments" / "yolov8s_v5_seed0.yaml"
        config = load_config(path)
        self.assertEqual(config["dataset"]["selection_split"], "validation")
        self.assertEqual(config["dataset"]["forbidden_selection_split"], "test")
        arguments = training_arguments(config, smoke=False)
        self.assertNotIn("test", arguments)
        self.assertEqual(arguments["freeze"], 0)
        self.assertEqual(arguments["optimizer"], "AdamW")
        self.assertEqual(arguments["lr0"], 0.0005)

    def test_smoke_overrides_are_small(self) -> None:
        path = PROJECT_ROOT / "configs" / "experiments" / "yolov8s_v5_seed0.yaml"
        config = load_config(path)
        arguments = training_arguments(config, smoke=True)
        self.assertEqual(arguments["epochs"], 1)
        self.assertEqual(arguments["imgsz"], 640)
        self.assertEqual(arguments["fraction"], 0.10)

    def test_p2_config_never_selects_test(self) -> None:
        path = PROJECT_ROOT / "configs" / "experiments" / "yolov8s_v5_seed0_p2.yaml"
        config = load_config(path)
        self.assertEqual(config["dataset"]["selection_split"], "validation")
        self.assertEqual(config["dataset"]["forbidden_selection_split"], "test")
        self.assertEqual(
            config["dataset"]["test_lock_sha256"],
            "6a9c275059591b7a1ae410b6a813e9c2041beac068801a37e44db5d432ebc1d3",
        )
        # Same dataset/init/hyperparameters as the baseline and the NWD
        # ablation, except architecture_yaml -- otherwise this would not be
        # a controlled comparison.
        baseline = load_config(
            PROJECT_ROOT / "configs" / "experiments" / "yolov8s_v5_seed0.yaml"
        )
        self.assertEqual(config["dataset"], baseline["dataset"])
        # Same hyperparameters except run "name" (must differ to avoid a
        # directory collision) and "batch" (halved from 4 to 2 -- measured
        # to avoid a near-certain CUDA OOM on this GPU with the extra P2
        # head's activation memory; see the config's own comment).
        excluded = {"name", "batch"}
        train_filtered = {k: v for k, v in config["train"].items() if k not in excluded}
        baseline_filtered = {
            k: v for k, v in baseline["train"].items() if k not in excluded
        }
        self.assertEqual(train_filtered, baseline_filtered)
        self.assertEqual(config["train"]["batch"], 2)
        self.assertEqual(config["model"]["weights"], baseline["model"]["weights"])
        self.assertIn("architecture_yaml", config["model"])

    def test_p2_architecture_yaml_adds_a_stride_4_head(self) -> None:
        path = (
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "architectures"
            / "yolov8s-p2-vietnam.yaml"
        )
        architecture = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(architecture["nc"], 5)
        detect_layer = architecture["head"][-1]
        # [[18, 21, 24, 27], 1, Detect, [nc]] -- four input scales, not three.
        self.assertEqual(len(detect_layer[0]), 4)
        self.assertEqual(detect_layer[2], "Detect")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch/ultralytics not installed")
    def test_p2_architecture_builds_with_a_stride_4_output_and_partial_transfer(
        self,
    ) -> None:
        from ultralytics import YOLO

        architecture_path = (
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "architectures"
            / "yolov8s-p2-vietnam.yaml"
        )
        model = YOLO(str(architecture_path))
        detect = model.model.model[-1]
        self.assertEqual(int(detect.nc), 5)
        strides = sorted(float(s) for s in detect.stride)
        self.assertEqual(strides, [4.0, 8.0, 16.0, 32.0])

        weights_path = (
            PROJECT_ROOT
            / "runs"
            / "detect"
            / "baseline"
            / "yolov8s_visdrone"
            / "weights"
            / "best.pt"
        )
        if weights_path.is_file():
            # Must not raise; architecture differs from the checkpoint, so
            # this is necessarily a partial (not 100%) transfer.
            model.load(str(weights_path))


if __name__ == "__main__":
    unittest.main()
