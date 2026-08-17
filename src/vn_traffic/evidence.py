"""Deterministic event-to-visual-evidence extraction for offline video."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import cv2

from .config import EvidenceConfig


EVIDENCE_SCHEMA_VERSION = 2
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_frame_sha256(frame: Any) -> str:
    """Hash the exact decoded BGR byte array before JPEG/video encoding."""
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


class EventEvidenceExporter:
    """Export raw keyframes and bounded clips in one sequential decode pass."""

    def __init__(
        self,
        config: EvidenceConfig,
        *,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    ):
        self.config = config
        self._capture_factory = capture_factory

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

    def _write_keyframe(
        self, frame: Any, frame_index: int, path: Path
    ) -> dict[str, Any]:
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
            "raw_bgr_sha256": _raw_frame_sha256(frame),
            "raw_shape": list(frame.shape),
            "raw_dtype": str(frame.dtype),
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

        source_sha256 = _sha256(source)
        records: list[dict[str, Any]] = []
        keyframes_by_frame: dict[
            int, list[tuple[dict[str, Any], Path]]
        ] = defaultdict(list)
        clips_by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
        clip_specs: list[dict[str, Any]] = []

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
                "source_video_sha256": source_sha256,
                "source_frame_index": frame_index,
                "source_timestamp_s": event.get("timestamp_s", frame_index / fps),
            }
            if event_type in self.config.keyframe_event_types:
                frames_dir.mkdir(parents=True, exist_ok=True)
                keyframe_path = frames_dir / f"{event_id}.jpg"
                keyframes_by_frame[frame_index].append((record, keyframe_path))
            if event_type in self.config.clip_event_types:
                clips_dir.mkdir(parents=True, exist_ok=True)
                start_frame = max(
                    0, frame_index - math.ceil(self.config.pre_event_s * fps)
                )
                end_frame = min(
                    frames_processed - 1,
                    frame_index + math.ceil(self.config.post_event_s * fps),
                )
                spec = {
                    "record": record,
                    "path": clips_dir / f"{event_id}.mp4",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "written": 0,
                    "writer": None,
                }
                clips_by_start[start_frame].append(spec)
                clip_specs.append(spec)
            records.append(record)

        capture = None
        active_clips: list[dict[str, Any]] = []
        try:
            if selected:
                capture = self._capture_factory(str(source))
                if not capture.isOpened():
                    raise ValueError(f"cannot reopen video source: {source}")
                for frame_index in range(frames_processed):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(
                            f"cannot decode evidence frame {frame_index} sequentially"
                        )

                    height, width = frame.shape[:2]
                    for spec in clips_by_start.get(frame_index, ()):
                        writer = cv2.VideoWriter(
                            str(spec["path"]),
                            cv2.VideoWriter_fourcc(*self.config.clip_codec),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            writer.release()
                            raise ValueError(
                                "cannot create evidence clip with codec "
                                f"{self.config.clip_codec}"
                            )
                        spec["writer"] = writer
                        active_clips.append(spec)

                    for record, path in keyframes_by_frame.get(frame_index, ()):
                        keyframe = self._write_keyframe(frame, frame_index, path)
                        keyframe["path"] = path.relative_to(run_dir).as_posix()
                        record["keyframe"] = keyframe

                    finished: list[dict[str, Any]] = []
                    for spec in active_clips:
                        spec["writer"].write(frame)
                        spec["written"] += 1
                        if frame_index == spec["end_frame"]:
                            spec["writer"].release()
                            spec["writer"] = None
                            finished.append(spec)
                    for spec in finished:
                        active_clips.remove(spec)
        finally:
            if capture is not None:
                capture.release()
            for spec in active_clips:
                writer = spec.get("writer")
                if writer is not None:
                    writer.release()

        for spec in clip_specs:
            expected = spec["end_frame"] - spec["start_frame"] + 1
            if spec["written"] != expected:
                raise ValueError(
                    f"evidence clip decoded {spec['written']} frames; expected {expected}"
                )
            path = spec["path"]
            spec["record"]["clip"] = {
                "path": path.relative_to(run_dir).as_posix(),
                "start_frame": spec["start_frame"],
                "end_frame": spec["end_frame"],
                "start_s": spec["start_frame"] / fps,
                "end_s": spec["end_frame"] / fps,
                "frame_count": spec["written"],
                "fps": fps,
                "sha256": _sha256(path),
            }

        temporary = manifest_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(manifest_path)
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "enabled": True,
            "extraction_mode": "sequential_second_pass",
            "source_video_sha256": source_sha256,
            "selected_events": len(records),
            "keyframes_written": sum("keyframe" in record for record in records),
            "clips_written": len(clip_specs),
            "manifest": manifest_path.name,
        }
