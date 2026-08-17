"""Deterministic event-to-visual-evidence extraction for offline video."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2

from .config import EvidenceConfig


EVIDENCE_SCHEMA_VERSION = 1
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EventEvidenceExporter:
    """Export raw keyframes and bounded clips referenced by event ID."""

    def __init__(self, config: EvidenceConfig):
        self.config = config

    def _load_events(self, path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_id = event.get("event_id")
                if not isinstance(event_id, str) or not _SAFE_EVENT_ID.fullmatch(
                    event_id
                ):
                    raise ValueError(
                        f"invalid event_id at {path.name}:{line_number}"
                    )
                if not isinstance(event.get("event_type"), str):
                    raise ValueError(
                        f"invalid event_type at {path.name}:{line_number}"
                    )
                if not isinstance(event.get("frame_index"), int):
                    raise ValueError(
                        f"invalid frame_index at {path.name}:{line_number}"
                    )
                if event_id in seen_event_ids:
                    raise ValueError(f"duplicate event_id in {path.name}: {event_id}")
                seen_event_ids.add(event_id)
                events.append(event)
        return events

    @staticmethod
    def _read_frame(capture: Any, frame_index: int) -> Any:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"cannot decode evidence frame {frame_index}")
        return frame

    def _write_keyframe(
        self,
        capture: Any,
        frame_index: int,
        path: Path,
    ) -> dict[str, Any]:
        frame = self._read_frame(capture, frame_index)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not ok:
            raise ValueError(f"cannot encode evidence keyframe {frame_index}")
        path.write_bytes(encoded.tobytes())
        height, width = frame.shape[:2]
        return {
            "path": path.as_posix(),
            "frame_index": frame_index,
            "width": width,
            "height": height,
            "sha256": _sha256(path),
        }

    def _write_clip(
        self,
        source: Path,
        start_frame: int,
        end_frame: int,
        fps: float,
        path: Path,
    ) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"cannot reopen video source: {source}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*self.config.clip_codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(
                f"cannot create evidence clip with codec {self.config.clip_codec}"
            )
        written = 0
        try:
            for _ in range(start_frame, end_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
            capture.release()
        if written != end_frame - start_frame + 1:
            raise ValueError(
                f"evidence clip decoded {written} frames; "
                f"expected {end_frame - start_frame + 1}"
            )
        return {
            "path": path.as_posix(),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": start_frame / fps,
            "end_s": end_frame / fps,
            "frame_count": written,
            "fps": fps,
            "sha256": _sha256(path),
        }

    def export(
        self,
        *,
        source: Path,
        events_path: Path,
        run_dir: Path,
        fps: float,
        frames_processed: int,
    ) -> dict[str, Any]:
        manifest_path = run_dir / "evidence.jsonl"
        frames_dir = run_dir / "evidence" / "frames"
        clips_dir = run_dir / "evidence" / "clips"
        events = self._load_events(events_path)
        selected = [
            event
            for event in events
            if event["event_type"] in self.config.keyframe_event_types
            or event["event_type"] in self.config.clip_event_types
        ]
        if frames_processed <= 0 and selected:
            raise ValueError("cannot extract evidence from an empty processed span")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"cannot reopen video source: {source}")
        records: list[dict[str, Any]] = []
        keyframes_written = 0
        clips_written = 0
        try:
            for event in selected:
                frame_index = event["frame_index"]
                if not 0 <= frame_index < frames_processed:
                    raise ValueError(
                        f"event {event['event_id']} frame is outside processed span"
                    )
                event_id = event["event_id"]
                event_type = event["event_type"]
                record: dict[str, Any] = {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "evidence_id": f"evidence-{event_id}",
                    "event_id": event_id,
                    "event_type": event_type,
                    "source_frame_index": frame_index,
                    "source_timestamp_s": event.get(
                        "timestamp_s", frame_index / fps
                    ),
                }
                if event_type in self.config.keyframe_event_types:
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    keyframe_path = frames_dir / f"{event_id}.jpg"
                    keyframe = self._write_keyframe(
                        capture,
                        frame_index,
                        keyframe_path,
                    )
                    keyframe["path"] = keyframe_path.relative_to(run_dir).as_posix()
                    record["keyframe"] = keyframe
                    keyframes_written += 1
                if event_type in self.config.clip_event_types:
                    clips_dir.mkdir(parents=True, exist_ok=True)
                    start_frame = max(
                        0,
                        frame_index - math.ceil(self.config.pre_event_s * fps),
                    )
                    end_frame = min(
                        frames_processed - 1,
                        frame_index + math.ceil(self.config.post_event_s * fps),
                    )
                    clip_path = clips_dir / f"{event_id}.mp4"
                    clip = self._write_clip(
                        source,
                        start_frame,
                        end_frame,
                        fps,
                        clip_path,
                    )
                    clip["path"] = clip_path.relative_to(run_dir).as_posix()
                    record["clip"] = clip
                    clips_written += 1
                records.append(record)
        finally:
            capture.release()

        temporary = manifest_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(manifest_path)
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "enabled": True,
            "selected_events": len(records),
            "keyframes_written": keyframes_written,
            "clips_written": clips_written,
            "manifest": manifest_path.name,
        }
