"""Offline-video runner and stable artifact writers."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import cv2

from .config import PipelineConfig
from .schemas import PerceptionResult, TRACK_CSV_FIELDS, TrackObservation


class PerceptionEngine(Protocol):
    def process(
        self, frame: Any, *, frame_index: int, timestamp_s: float
    ) -> PerceptionResult: ...


class EventProcessor(Protocol):
    def process(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        tracks: tuple[TrackObservation, ...],
    ) -> Iterable[dict[str, Any]]: ...


class NoEvents:
    def process(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        tracks: tuple[TrackObservation, ...],
    ) -> Iterable[dict[str, Any]]:
        return ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("run"):
            try:
                numbers.append(int(child.name[3:]))
            except ValueError:
                continue
    run_dir = root / f"run{max(numbers, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        perception: PerceptionEngine,
        event_processor: EventProcessor | None = None,
    ):
        self.config = config
        self.perception = perception
        self.event_processor = event_processor or NoEvents()

    def run(self) -> Path:
        if not self.config.source.is_file():
            raise FileNotFoundError(f"video source not found: {self.config.source}")

        capture = cv2.VideoCapture(str(self.config.source))
        if not capture.isOpened():
            raise ValueError(f"cannot open video source: {self.config.source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = self.config.fallback_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("video source reports an invalid frame size")

        run_dir = next_run_directory(self.config.output_root)
        video_path = run_dir / "annotated.mp4"
        tracks_path = run_dir / "tracks.csv"
        events_path = run_dir / "events.jsonl"
        metadata_path = run_dir / "run.json"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*self.config.codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(
                f"cannot create video writer with codec {self.config.codec}"
            )

        started_at = utc_now()
        started_clock = time.perf_counter()
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "running",
            "started_at": started_at,
            "source": str(self.config.source),
            "model": str(self.config.model),
            "config": (
                str(self.config.config_path) if self.config.config_path else None
            ),
            "video": {
                "fps": fps,
                "width": width,
                "height": height,
                "codec": self.config.codec,
            },
            "perception": {
                "imgsz": self.config.imgsz,
                "confidence": self.config.confidence,
                "iou": self.config.iou,
                "device": self.config.device,
                "tracker": self.config.tracker,
            },
            "outputs": {
                "annotated_video": "annotated.mp4",
                "tracks": "tracks.csv",
                "events": "events.jsonl",
            },
        }
        write_json_atomic(metadata_path, metadata)

        frame_count = 0
        track_count = 0
        event_count = 0
        try:
            with tracks_path.open("w", newline="", encoding="utf-8") as tracks_file, (
                events_path.open("w", encoding="utf-8")
            ) as events_file:
                track_writer = csv.DictWriter(
                    tracks_file, fieldnames=list(TRACK_CSV_FIELDS)
                )
                track_writer.writeheader()
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index = frame_count
                    timestamp_s = frame_index / fps
                    result = self.perception.process(
                        frame,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                    )
                    annotated = result.annotated_frame
                    if annotated.shape[1] != width or annotated.shape[0] != height:
                        annotated = cv2.resize(annotated, (width, height))
                    writer.write(annotated)
                    for track in result.tracks:
                        track_writer.writerow(track.to_dict())
                        track_count += 1
                    events = self.event_processor.process(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        tracks=result.tracks,
                    )
                    for event in events:
                        events_file.write(
                            json.dumps(event, ensure_ascii=False) + "\n"
                        )
                        event_count += 1
                    frame_count += 1
                    if (
                        self.config.max_frames is not None
                        and frame_count >= self.config.max_frames
                    ):
                        break

            elapsed_s = time.perf_counter() - started_clock
            metadata.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "frames_processed": frame_count,
                    "track_rows": track_count,
                    "events_written": event_count,
                    "elapsed_s": elapsed_s,
                    "processing_fps": frame_count / elapsed_s if elapsed_s else None,
                }
            )
            write_json_atomic(metadata_path, metadata)
            return run_dir
        except Exception as error:
            metadata.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "frames_processed": frame_count,
                    "track_rows": track_count,
                    "events_written": event_count,
                }
            )
            write_json_atomic(metadata_path, metadata)
            raise
        finally:
            capture.release()
            writer.release()
