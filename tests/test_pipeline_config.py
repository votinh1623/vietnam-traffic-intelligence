from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.config import load_pipeline_config  # noqa: E402


class PipelineConfigTests(unittest.TestCase):
    def test_loads_absolute_paths_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input.mp4"
            model = root / "model.pt"
            config_path = root / "pipeline.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "schema_version: 1",
                        f"source: '{source.as_posix()}'",
                        f"model: '{model.as_posix()}'",
                        "perception:",
                        "  imgsz: 640",
                        "  confidence: 0.25",
                        "  device: cpu",
                        "video:",
                        "  codec: mp4v",
                        "output:",
                        f"  root: '{(root / 'outputs').as_posix()}'",
                    )
                ),
                encoding="utf-8",
            )
            config = load_pipeline_config(config_path)
            self.assertEqual(config.source, source.resolve())
            self.assertEqual(config.model, model.resolve())
            self.assertEqual(config.imgsz, 640)
            self.assertEqual(config.confidence, 0.25)
            self.assertEqual(config.device, "cpu")
            self.assertTrue(config.analytics.enabled)

    def test_resolves_repository_tracker_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "perception:\n  tracker: bytetrack_custom.yaml\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertEqual(
                Path(config.tracker), (PROJECT_ROOT / "bytetrack_custom.yaml").resolve()
            )

    def test_rejects_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text("schema_version: 1\nmodel: model.pt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source is required"):
                load_pipeline_config(path)

    def test_rejects_invalid_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "perception:\n  confidence: 1.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "confidence"):
                load_pipeline_config(path)


if __name__ == "__main__":
    unittest.main()
