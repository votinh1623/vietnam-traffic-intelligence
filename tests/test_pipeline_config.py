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
                        "  max_det: 1000",
                        "  show_labels: false",
                        "  show_confidence: false",
                        "  line_width: 1",
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
            self.assertEqual(config.max_det, 1000)
            self.assertFalse(config.show_labels)
            self.assertFalse(config.show_confidence)
            self.assertEqual(config.line_width, 1)
            self.assertEqual(config.device, "cpu")
            self.assertTrue(config.analytics.enabled)
            self.assertIn("car", config.analytics.included_classes)
            self.assertNotIn("pedestrian", config.analytics.included_classes)
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

    def test_loads_stillness_heatmap_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "stillness_heatmap:\n"
                "  enabled: true\n"
                "  downscale: 2\n"
                "  cell_px: 4\n"
                "  motion_threshold: 0.8\n"
                "  texture_percentile: 85\n"
                "  alpha_max: 0.3\n"
                "  smoothing_decay: 0.6\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertTrue(config.stillness_heatmap.enabled)
            self.assertEqual(config.stillness_heatmap.downscale, 2)
            self.assertEqual(config.stillness_heatmap.cell_px, 4)
            self.assertEqual(config.stillness_heatmap.motion_threshold, 0.8)
            self.assertEqual(config.stillness_heatmap.texture_percentile, 85)
            self.assertEqual(config.stillness_heatmap.alpha_max, 0.3)
            self.assertEqual(config.stillness_heatmap.smoothing_decay, 0.6)

    def test_stillness_heatmap_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text("source: input.mp4\nmodel: model.pt\n", encoding="utf-8")
            config = load_pipeline_config(path)
            self.assertFalse(config.stillness_heatmap.enabled)

    def test_rejects_invalid_stillness_heatmap_percentile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "stillness_heatmap:\n  texture_percentile: 150\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "texture_percentile"):
                load_pipeline_config(path)

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

    def test_uav_motion_mode_defaults_roi_to_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  mode: uav_motion\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertEqual(config.analytics.analytics_mode, "uav_motion")
            self.assertEqual(
                config.analytics.roi_polygon,
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            )
            self.assertEqual(
                config.analytics.counting_line, ((0.0, 0.5), (1.0, 0.5))
            )

    def test_uav_motion_mode_still_honors_explicit_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  mode: uav_motion\n  roi_polygon:\n"
                "    - [0.1, 0.1]\n    - [0.9, 0.1]\n"
                "    - [0.9, 0.9]\n    - [0.1, 0.9]\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertEqual(
                config.analytics.roi_polygon,
                ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
            )

    def test_fixed_camera_mode_is_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text("source: input.mp4\nmodel: model.pt\n", encoding="utf-8")
            config = load_pipeline_config(path)
            self.assertEqual(config.analytics.analytics_mode, "fixed_camera")

    def test_loads_gmc_settings_under_uav_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  mode: uav_motion\n"
                "  gmc_enabled: true\n  gmc_downscale: 2\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertTrue(config.analytics.gmc_enabled)
            self.assertEqual(config.analytics.gmc_downscale, 2)

    def test_rejects_stillness_enabled_under_uav_motion_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  mode: uav_motion\n"
                "  stillness_enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stillness_enabled"):
                load_pipeline_config(path)

    def test_rejects_gmc_enabled_without_uav_motion_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  gmc_enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "gmc_enabled"):
                load_pipeline_config(path)

    def test_rejects_unknown_analytics_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "analytics:\n  mode: bev_calibrated\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "analytics.mode"):
                load_pipeline_config(path)

    def test_resolves_repository_tracker_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.yaml"
            path.write_text(
                "source: input.mp4\nmodel: model.pt\n"
                "perception:\n  tracker: botsort_reid_lowprox.yaml\n",
                encoding="utf-8",
            )
            config = load_pipeline_config(path)
            self.assertEqual(
                Path(config.tracker),
                (PROJECT_ROOT / "botsort_reid_lowprox.yaml").resolve(),
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
