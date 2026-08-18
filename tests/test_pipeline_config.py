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
            self.assertEqual(config.analytics.occupancy_grid_size_px, 1)
            self.assertFalse(config.evidence.enabled)

    def test_loads_evidence_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "evidence:\n"
                "  enabled: true\n"
                "  keyframe_event_types: [line_crossing]\n"
                "  clip_event_types: []\n"
                "  pre_event_s: 1.5\n"
                "  post_event_s: 2.5\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertTrue(config.evidence.enabled)
            self.assertEqual(config.evidence.keyframe_event_types, ("line_crossing",))
            self.assertEqual(config.evidence.clip_event_types, ())
            self.assertEqual(config.evidence.pre_event_s, 1.5)
            self.assertEqual(config.evidence.post_event_s, 2.5)

    def test_loads_prolonged_stop_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  abnormal:\n"
                "    prolonged_stop_enabled: true\n"
                "    prolonged_stop_classes: [car, bus]\n"
                "    prolonged_stop_max_speed_px_s: 3\n"
                "    prolonged_stop_release_speed_px_s: 7\n"
                "    prolonged_stop_min_duration_s: 4\n"
                "    prolonged_stop_max_gap_s: 0.5\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertTrue(config.analytics.prolonged_stop_enabled)
            self.assertEqual(config.analytics.prolonged_stop_classes, ("car", "bus"))
            self.assertEqual(config.analytics.prolonged_stop_min_duration_s, 4.0)
            self.assertEqual(config.analytics.prolonged_stop_max_gap_s, 0.5)

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

    def test_rejects_legacy_summed_occupancy_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  congestion:\n    dense_enter_occupancy: 0.3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy summed-occupancy"):
                load_pipeline_config(path)

    def test_rejects_inverted_prolonged_stop_speed_hysteresis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  abnormal:\n"
                "    prolonged_stop_max_speed_px_s: 10\n"
                "    prolonged_stop_release_speed_px_s: 5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "release_speed"):
                load_pipeline_config(path)


if __name__ == "__main__":
    unittest.main()
