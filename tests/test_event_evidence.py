from __future__ import annotations

import hashlib
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
from vn_traffic.evidence import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EventEvidenceExporter,
)


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


class NoSeekCapture:
    """Delegate sequential reads but fail if the exporter attempts any seek."""

    def __init__(self, source: str):
        self.capture = cv2.VideoCapture(source)

    def isOpened(self):  # noqa: N802 - mirrors OpenCV
        return self.capture.isOpened()

    def read(self):
        return self.capture.read()

    def set(self, *_args):
        raise AssertionError("evidence exporter must not seek")

    def release(self):
        self.capture.release()


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
            self.assertEqual(summary["schema_version"], EVIDENCE_SCHEMA_VERSION)
            self.assertEqual(summary["extraction_mode"], "sequential_second_pass")
            self.assertEqual(len(summary["source_video_sha256"]), 64)
            self.assertIn("keyframe", records[0])
            self.assertNotIn("clip", records[0])
            self.assertEqual(records[1]["clip"]["frame_count"], 5)
            self.assertEqual(records[1]["clip"]["start_frame"], 1)
            self.assertEqual(records[1]["clip"]["end_frame"], 5)
            for record in records:
                keyframe = run_dir / record["keyframe"]["path"]
                self.assertTrue(keyframe.is_file())
                self.assertEqual(len(record["keyframe"]["sha256"]), 64)
                self.assertEqual(len(record["keyframe"]["raw_bgr_sha256"]), 64)
                self.assertEqual(record["keyframe"]["raw_shape"], [48, 64, 3])
                self.assertEqual(record["keyframe"]["raw_dtype"], "uint8")
                self.assertEqual(
                    record["source_video_sha256"], summary["source_video_sha256"]
                )
            first_keyframe = cv2.imread(str(run_dir / records[0]["keyframe"]["path"]))
            self.assertGreater(float(first_keyframe.mean()), 55.0)
            self.assertLess(float(first_keyframe.mean()), 65.0)
            clip = run_dir / records[1]["clip"]["path"]
            self.assertTrue(clip.is_file())
            self.assertEqual(len(records[1]["clip"]["sha256"]), 64)

    def test_never_seeks_during_evidence_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.avi"
            events_path = run_dir / "events.jsonl"
            create_video(source)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "event-000001",
                        "event_type": "congestion_transition",
                        "frame_index": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            exporter = EventEvidenceExporter(
                EvidenceConfig(
                    enabled=True,
                    pre_event_s=0.4,
                    post_event_s=0.4,
                    clip_codec="mp4v",
                ),
                capture_factory=NoSeekCapture,
            )

            summary = exporter.export(
                source=source,
                events_path=events_path,
                run_dir=run_dir,
                fps=5.0,
                frames_processed=6,
            )

            self.assertEqual(summary["keyframes_written"], 1)
            self.assertEqual(summary["clips_written"], 1)

    def test_overlapping_clips_are_clamped_to_processed_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.avi"
            events_path = run_dir / "events.jsonl"
            create_video(source)
            events = (
                {
                    "event_id": "event-start",
                    "event_type": "congestion_transition",
                    "frame_index": 1,
                },
                {
                    "event_id": "event-end",
                    "event_type": "congestion_transition",
                    "frame_index": 4,
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

            exporter.export(
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
            self.assertEqual(
                (records[0]["clip"]["start_frame"], records[0]["clip"]["end_frame"]),
                (0, 3),
            )
            self.assertEqual(records[0]["clip"]["frame_count"], 4)
            self.assertEqual(
                (records[1]["clip"]["start_frame"], records[1]["clip"]["end_frame"]),
                (2, 5),
            )
            self.assertEqual(records[1]["clip"]["frame_count"], 4)

            capture = cv2.VideoCapture(str(source))
            expected_hashes = {}
            for frame_index in range(6):
                ok, frame = capture.read()
                self.assertTrue(ok)
                if frame_index in (1, 4):
                    expected_hashes[frame_index] = hashlib.sha256(
                        frame.tobytes(order="C")
                    ).hexdigest()
            capture.release()
            self.assertEqual(
                records[0]["keyframe"]["raw_bgr_sha256"], expected_hashes[1]
            )
            self.assertEqual(
                records[1]["keyframe"]["raw_bgr_sha256"], expected_hashes[4]
            )

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


def write_tracks_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = [
        "frame_index", "timestamp_s", "track_id", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def measurement_manifest(roi_polygon) -> dict:
    return {
        "schema_version": 1,
        "scene_id": "fixture",
        "camera": {"motion": "unknown"},
        "measurement": {
            "roi_polygon": roi_polygon,
            "ignore_regions": [],
            "regions": [],
            "counting_lines": [],
        },
        "provenance": {
            "author": "test",
            "created_at": "2026-08-27T00:00:00+00:00",
            "source_sha256": "0" * 64,
        },
    }


class MultiViewEvidenceTests(unittest.TestCase):
    def _export(
        self,
        *,
        events: tuple[dict, ...],
        tracks: list[dict] | None = None,
        manifest: dict | None = None,
        multi_view_enabled: bool = True,
    ):
        # Deliberately not a TemporaryDirectory context manager: callers
        # inspect files under run_dir after this returns, so the directory
        # must outlive this method. Test-only leak onto the OS temp dir.
        run_dir = Path(tempfile.mkdtemp())
        source = run_dir / "source.avi"
        events_path = run_dir / "events.jsonl"
        create_video(source)
        events_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        tracks_path = None
        if tracks is not None:
            tracks_path = run_dir / "tracks.csv"
            write_tracks_csv(tracks_path, tracks)
        exporter = EventEvidenceExporter(
            EvidenceConfig(
                enabled=True,
                keyframe_event_types=(
                    "line_crossing",
                    "congestion_transition",
                ),
                clip_event_types=(),
                multi_view_enabled=multi_view_enabled,
            )
        )
        summary = exporter.export(
            source=source,
            events_path=events_path,
            run_dir=run_dir,
            fps=5.0,
            frames_processed=6,
            measurement_manifest=manifest,
            tracks_path=tracks_path,
        )
        records = [
            json.loads(line)
            for line in (run_dir / "evidence.jsonl").read_text().splitlines()
        ]
        return run_dir, summary, records

    def test_event_crop_written_when_track_bbox_resolvable(self) -> None:
        run_dir, summary, records = self._export(
            events=(
                {
                    "event_id": "event-000001",
                    "event_type": "line_crossing",
                    "frame_index": 2,
                    "timestamp_s": 0.4,
                    "track_id": 7,
                },
            ),
            tracks=[
                {
                    "frame_index": 2, "timestamp_s": 0.4, "track_id": 7,
                    "class_id": 1, "class_name": "car", "confidence": 0.9,
                    "x1": 10, "y1": 10, "x2": 30, "y2": 30,
                }
            ],
            manifest=measurement_manifest([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        self.assertEqual(summary["event_crops_written"], 1)
        self.assertEqual(summary["roi_crops_written"], 0)
        record = records[0]
        self.assertIn("event_crop", record)
        self.assertNotIn("roi_crop", record)
        crop = record["event_crop"]
        self.assertEqual(len(crop["bbox_px"]), 4)
        # Padded around the 20x20 track bbox -- strictly smaller than the
        # full 64x48 frame.
        self.assertLess(crop["width"], 64)
        self.assertLess(crop["height"], 48)
        self.assertTrue((run_dir / crop["path"]).is_file())

    def test_roi_crop_fallback_when_event_has_no_track_id(self) -> None:
        run_dir, summary, records = self._export(
            events=(
                {
                    "event_id": "event-000002",
                    "event_type": "congestion_transition",
                    "frame_index": 3,
                    "timestamp_s": 0.6,
                },
            ),
            tracks=[],
            manifest=measurement_manifest([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]),
        )
        self.assertEqual(summary["event_crops_written"], 0)
        self.assertEqual(summary["roi_crops_written"], 1)
        record = records[0]
        self.assertIn("roi_crop", record)
        self.assertNotIn("event_crop", record)
        self.assertTrue((run_dir / record["roi_crop"]["path"]).is_file())

    def test_no_crop_when_no_manifest_and_no_resolvable_track(self) -> None:
        run_dir, summary, records = self._export(
            events=(
                {
                    "event_id": "event-000003",
                    "event_type": "congestion_transition",
                    "frame_index": 3,
                    "timestamp_s": 0.6,
                },
            ),
            tracks=None,
            manifest=None,
        )
        self.assertEqual(summary["event_crops_written"], 0)
        self.assertEqual(summary["roi_crops_written"], 0)
        record = records[0]
        self.assertIn("keyframe", record)
        self.assertNotIn("roi_crop", record)
        self.assertNotIn("event_crop", record)

    def test_multi_view_disabled_by_default_produces_only_keyframe(self) -> None:
        run_dir, summary, records = self._export(
            events=(
                {
                    "event_id": "event-000004",
                    "event_type": "line_crossing",
                    "frame_index": 2,
                    "timestamp_s": 0.4,
                    "track_id": 7,
                },
            ),
            tracks=[
                {
                    "frame_index": 2, "timestamp_s": 0.4, "track_id": 7,
                    "class_id": 1, "class_name": "car", "confidence": 0.9,
                    "x1": 10, "y1": 10, "x2": 30, "y2": 30,
                }
            ],
            manifest=measurement_manifest([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            multi_view_enabled=False,
        )
        self.assertFalse(summary["multi_view_enabled"])
        self.assertEqual(summary["event_crops_written"], 0)
        self.assertEqual(summary["roi_crops_written"], 0)
        self.assertNotIn("roi_crop", records[0])
        self.assertNotIn("event_crop", records[0])


if __name__ == "__main__":
    unittest.main()
