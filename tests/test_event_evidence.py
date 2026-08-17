from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.config import EvidenceConfig  # noqa: E402
from vn_traffic.evidence import EventEvidenceExporter  # noqa: E402


def create_video(path: Path, frame_count: int = 6, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("MJPG fixture writer is unavailable")
    for index in range(frame_count):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()


class EventEvidenceExporterTests(unittest.TestCase):
    def test_exports_keyframes_and_only_configured_clip_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.avi"
            events_path = run_dir / "events.jsonl"
            create_video(source)
            events = (
                {
                    "event_id": "event-000001",
                    "event_type": "line_crossing",
                    "frame_index": 2,
                    "timestamp_s": 0.4,
                },
                {
                    "event_id": "event-000002",
                    "event_type": "congestion_transition",
                    "frame_index": 3,
                    "timestamp_s": 0.6,
                },
            )
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            exporter = EventEvidenceExporter(
                EvidenceConfig(
                    enabled=True,
                    pre_event_s=0.4,
                    post_event_s=0.4,
                    clip_codec="mp4v",
                )
            )

            summary = exporter.export(
                source=source,
                events_path=events_path,
                run_dir=run_dir,
                fps=5.0,
                frames_processed=6,
            )

            records = [
                json.loads(line)
                for line in (run_dir / "evidence.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["selected_events"], 2)
            self.assertEqual(summary["keyframes_written"], 2)
            self.assertEqual(summary["clips_written"], 1)
            self.assertIn("keyframe", records[0])
            self.assertNotIn("clip", records[0])
            self.assertEqual(records[1]["clip"]["frame_count"], 5)
            self.assertEqual(records[1]["clip"]["start_frame"], 1)
            self.assertEqual(records[1]["clip"]["end_frame"], 5)
            for record in records:
                keyframe = run_dir / record["keyframe"]["path"]
                self.assertTrue(keyframe.is_file())
                self.assertEqual(len(record["keyframe"]["sha256"]), 64)
            first_keyframe = cv2.imread(str(run_dir / records[0]["keyframe"]["path"]))
            self.assertGreater(float(first_keyframe.mean()), 55.0)
            self.assertLess(float(first_keyframe.mean()), 65.0)
            clip = run_dir / records[1]["clip"]["path"]
            self.assertTrue(clip.is_file())
            self.assertEqual(len(records[1]["clip"]["sha256"]), 64)

    def test_empty_events_create_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.avi"
            events_path = run_dir / "events.jsonl"
            create_video(source, frame_count=1)
            events_path.write_text("", encoding="utf-8")
            summary = EventEvidenceExporter(EvidenceConfig(enabled=True)).export(
                source=source,
                events_path=events_path,
                run_dir=run_dir,
                fps=5.0,
                frames_processed=1,
            )
            self.assertEqual(summary["selected_events"], 0)
            self.assertEqual((run_dir / "evidence.jsonl").read_text(), "")

    def test_rejects_unsafe_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.avi"
            events_path = run_dir / "events.jsonl"
            create_video(source, frame_count=1)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "../escape",
                        "event_type": "line_crossing",
                        "frame_index": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid event_id"):
                EventEvidenceExporter(EvidenceConfig(enabled=True)).export(
                    source=source,
                    events_path=events_path,
                    run_dir=run_dir,
                    fps=5.0,
                    frames_processed=1,
                )

    def test_rejects_duplicate_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.avi"
            events_path = run_dir / "events.jsonl"
            create_video(source, frame_count=1)
            event = {
                "event_id": "event-000001",
                "event_type": "line_crossing",
                "frame_index": 0,
            }
            events_path.write_text(
                json.dumps(event) + "\n" + json.dumps(event) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate event_id"):
                EventEvidenceExporter(EvidenceConfig(enabled=True)).export(
                    source=source,
                    events_path=events_path,
                    run_dir=run_dir,
                    fps=5.0,
                    frames_processed=1,
                )


if __name__ == "__main__":
    unittest.main()
