from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.config import PipelineConfig  # noqa: E402
from vn_traffic.reasoning.freeze import file_sha256  # noqa: E402
from vn_traffic.runner import PipelineRunner  # noqa: E402
from vn_traffic.schemas import (  # noqa: E402
    ANALYTICS_SCHEMA_VERSION,
    PerceptionResult,
    TrackObservation,
)


class FakePerception:
    def process(self, frame, *, frame_index: int, timestamp_s: float):
        annotated = frame.copy()
        cv2.rectangle(annotated, (4, 4), (20, 20), (0, 255, 0), 1)
        track = TrackObservation(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            track_id=7,
            class_id=1,
            class_name="car",
            confidence=0.9,
            x1=4.0,
            y1=4.0,
            x2=20.0,
            y2=20.0,
        )
        return PerceptionResult(annotated_frame=annotated, tracks=(track,))


class FailingPerception:
    def process(self, frame, *, frame_index: int, timestamp_s: float):
        raise RuntimeError("intentional fixture failure")


def create_fixture_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("MJPG fixture writer is unavailable")
    for index in range(frame_count):
        frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class PipelineRunnerTests(unittest.TestCase):
    def test_creates_stable_stage_one_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            model = root / "placeholder.pt"
            create_fixture_video(source)
            model.write_bytes(b"fake model for dependency-injected test")
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=model,
                output_root=root / "outputs",
                imgsz=64,
                device="cpu",
            )

            run_dir = PipelineRunner(config, FakePerception()).run()

            self.assertEqual(run_dir.name, "run1")
            for name in (
                "annotated.mp4",
                "tracks.csv",
                "events.jsonl",
                "analytics.csv",
                "summary.json",
                "evidence.jsonl",
                "run.json",
                "latest_frame.jpg",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertEqual(list(run_dir.glob("latest_frame*.tmp*")), [])
            last_frame = cv2.imread(str(run_dir / "latest_frame.jpg"))
            self.assertIsNotNone(last_frame)
            self.assertEqual(last_frame.shape[:2], (48, 64))
            with (run_dir / "tracks.csv").open(newline="", encoding="utf-8") as stream:
                tracks = list(csv.DictReader(stream))
            self.assertEqual(len(tracks), 3)
            self.assertEqual(tracks[0]["track_id"], "7")
            self.assertEqual(tracks[0]["class_name"], "car")
            self.assertEqual((run_dir / "events.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual(
                (run_dir / "evidence.jsonl").read_text(encoding="utf-8"),
                "",
            )
            analytics_lines = (run_dir / "analytics.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(analytics_lines), 1)
            self.assertIn("bbox_union_occupancy", analytics_lines[0])
            self.assertNotIn(",occupancy,", f",{analytics_lines[0]},")
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["analytics_enabled"])
            self.assertEqual(summary["schema_version"], ANALYTICS_SCHEMA_VERSION)
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["frames_processed"], 3)
            self.assertEqual(metadata["track_rows"], 3)
            self.assertEqual(metadata["events_written"], 0)
            self.assertEqual(
                metadata["analytics_schema_version"],
                ANALYTICS_SCHEMA_VERSION,
            )
            self.assertGreater(metadata["processing_fps"], 0)
            self.assertFalse(metadata["evidence"]["enabled"])
            provenance = metadata["provenance"]
            self.assertIn("commit", provenance["git"])
            self.assertIn("python_version", provenance["environment"])
            self.assertEqual(len(provenance["source_sha256"]), 64)
            self.assertEqual(len(provenance["model_sha256"]), 64)
            self.assertIsNone(provenance["config_sha256"])
            self.assertIsNone(provenance["tracker_sha256"])

            capture = cv2.VideoCapture(str(run_dir / "annotated.mp4"))
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            self.assertEqual(output_frames, 3)

    def test_live_frame_write_failure_does_not_crash_the_run(self) -> None:
        # Regression test: on Windows, os.replace() can raise WinError 5
        # (Access is denied) if latest_frame.jpg is momentarily locked by
        # another process (antivirus/indexer scanning the new file). The
        # live-frame preview is a dashboard convenience, not a required
        # artifact, so this must never crash the pipeline run.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            model = root / "placeholder.pt"
            create_fixture_video(source)
            model.write_bytes(b"fake model for dependency-injected test")
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=model,
                output_root=root / "outputs",
                imgsz=64,
                device="cpu",
            )

            real_replace = Path.replace

            def flaky_replace(self, target):
                if self.name.startswith("latest_frame"):
                    raise PermissionError(5, "Access is denied")
                return real_replace(self, target)

            with patch.object(Path, "replace", flaky_replace):
                run_dir = PipelineRunner(config, FakePerception()).run()

            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertFalse((run_dir / "latest_frame.jpg").exists())

    def test_run_json_write_retries_transient_permission_error(self) -> None:
        # Regression test: run87 crashed on a real machine when
        # run.json.tmp -> run.json hit WinError 5 (Access is denied) from a
        # transient antivirus/indexer lock -- unlike latest_frame.jpg (a
        # dashboard convenience that can be silently skipped),
        # write_json_atomic must retry before giving up, since run.json is
        # the run's own status/provenance record.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            model = root / "placeholder.pt"
            create_fixture_video(source)
            model.write_bytes(b"fake model for dependency-injected test")
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=model,
                output_root=root / "outputs",
                imgsz=64,
                device="cpu",
            )

            real_replace = Path.replace
            remaining_failures = [2]

            def flaky_replace(self, target):
                if self.name == "run.json.tmp" and remaining_failures[0] > 0:
                    remaining_failures[0] -= 1
                    raise PermissionError(5, "Access is denied")
                return real_replace(self, target)

            with patch.object(Path, "replace", flaky_replace), patch(
                "vn_traffic.runner.time.sleep"
            ):
                run_dir = PipelineRunner(config, FakePerception()).run()

            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(remaining_failures[0], 0)

    def test_records_failed_run_without_hiding_the_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            create_fixture_video(source, frame_count=1)
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=root / "placeholder.pt",
                output_root=root / "outputs",
            )

            with self.assertRaisesRegex(RuntimeError, "intentional fixture failure"):
                PipelineRunner(config, FailingPerception()).run()

            metadata = json.loads(
                (root / "outputs" / "run1" / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("intentional fixture failure", metadata["error"])

    def _measurement_manifest_payload(self, *, source_sha256: str) -> dict:
        return {
            "schema_version": 1,
            "scene_id": "fixture_scene",
            "camera": {"motion": "static"},
            "measurement": {
                "roi_polygon": None,
                "ignore_regions": [],
                "regions": [],
                "counting_lines": [],
            },
            "provenance": {
                "author": "test",
                "created_at": "2026-08-27T00:00:00+00:00",
                "source_sha256": source_sha256,
            },
        }

    def test_measurement_manifest_is_loaded_and_recorded_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            model = root / "placeholder.pt"
            create_fixture_video(source)
            model.write_bytes(b"fake model for dependency-injected test")
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    self._measurement_manifest_payload(
                        source_sha256=file_sha256(source)
                    )
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=model,
                output_root=root / "outputs",
                imgsz=64,
                device="cpu",
                measurement_manifest=manifest_path,
            )

            run_dir = PipelineRunner(config, FakePerception()).run()

            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(
                metadata["measurement_manifest"]["scene_id"], "fixture_scene"
            )

    def test_measurement_manifest_absent_does_not_fail_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            model = root / "placeholder.pt"
            create_fixture_video(source)
            model.write_bytes(b"fake model for dependency-injected test")
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=model,
                output_root=root / "outputs",
                imgsz=64,
                device="cpu",
            )

            run_dir = PipelineRunner(config, FakePerception()).run()

            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertIsNone(metadata["measurement_manifest"])

    def test_measurement_manifest_wrong_source_hash_fails_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.avi"
            model = root / "placeholder.pt"
            create_fixture_video(source)
            model.write_bytes(b"fake model for dependency-injected test")
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    self._measurement_manifest_payload(source_sha256="0" * 64)
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                schema_version=1,
                source=source,
                model=model,
                output_root=root / "outputs",
                imgsz=64,
                device="cpu",
                measurement_manifest=manifest_path,
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                PipelineRunner(config, FakePerception()).run()

            metadata = json.loads(
                (root / "outputs" / "run1" / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")


if __name__ == "__main__":
    unittest.main()
