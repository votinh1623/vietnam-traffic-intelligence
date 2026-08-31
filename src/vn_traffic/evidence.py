"""Deterministic event-to-visual-evidence extraction for offline video."""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import cv2

from .config import EvidenceConfig


EVIDENCE_SCHEMA_VERSION = 3
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


def _roi_bbox_px(
    roi_polygon: list[list[float]] | None, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Pixel bounding box of a normalized-[0,1] ROI polygon, or None (full
    frame -- no manifest, or manifest declares roi_polygon: null)."""
    if not roi_polygon:
        return None
    xs = [point[0] for point in roi_polygon]
    ys = [point[1] for point in roi_polygon]
    x1 = max(0, min(width - 1, round(min(xs) * width)))
    y1 = max(0, min(height - 1, round(min(ys) * height)))
    x2 = max(x1 + 1, min(width, round(max(xs) * width)))
    y2 = max(y1 + 1, min(height, round(max(ys) * height)))
    return (x1, y1, x2, y2)


def _padded_bbox_px(
    x1: float, y1: float, x2: float, y2: float,
    *, padding_ratio: float, width: int, height: int,
) -> tuple[int, int, int, int]:
    """Track bbox (already pixel-space) padded by `padding_ratio` on each
    side and clamped to frame bounds."""
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    px1 = max(0, min(width - 1, round(x1 - pad_x)))
    py1 = max(0, min(height - 1, round(y1 - pad_y)))
    px2 = max(px1 + 1, min(width, round(x2 + pad_x)))
    py2 = max(py1 + 1, min(height, round(y2 + pad_y)))
    return (px1, py1, px2, py2)


def _load_track_bboxes(tracks_path: Path) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Index tracks.csv by (frame_index, track_id) -> (x1, y1, x2, y2), for
    resolving an event's own bbox when writing its event_crop."""
    bboxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    if not tracks_path.is_file():
        return bboxes
    with tracks_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("track_id"):
                continue
            key = (int(row["frame_index"]), int(row["track_id"]))
            bboxes[key] = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
    return bboxes


class EventEvidenceExporter:
    """Export raw keyframes, ROI/event crops, and bounded clips in one
    sequential decode pass."""

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

    def _write_image_artifact(
        self,
        frame: Any,
        frame_index: int,
        path: Path,
        *,
        bbox_px: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        """Encode `frame` (or the `bbox_px` crop of it) as a JPEG artifact
        record. `bbox_px`, when given, is the crop region within the source
        frame -- the artifact's own width/height are the crop's, not the
        source frame's, but bbox_px preserves where it came from."""
        crop = frame
        if bbox_px is not None:
            x1, y1, x2, y2 = bbox_px
            crop = frame[y1:y2, x1:x2]
        ok, encoded = cv2.imencode(
            ".jpg",
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not ok:
            raise ValueError(f"cannot encode evidence image at frame {frame_index}")
        path.write_bytes(encoded.tobytes())
        height, width = crop.shape[:2]
        artifact: dict[str, Any] = {
            "path": path.as_posix(),
            "frame_index": frame_index,
            "width": width,
            "height": height,
            "raw_bgr_sha256": _raw_frame_sha256(crop),
            "raw_shape": list(crop.shape),
            "raw_dtype": str(crop.dtype),
            "sha256": _sha256(path),
        }
        if bbox_px is not None:
            artifact["bbox_px"] = list(bbox_px)
        return artifact

    def export(
        self,
        *,
        source: Path,
        events_path: Path,
        run_dir: Path,
        fps: float,
        frames_processed: int,
        measurement_manifest: dict[str, Any] | None = None,
        tracks_path: Path | None = None,
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

        multi_view = self.config.multi_view_enabled
        roi_polygon = (
            measurement_manifest["measurement"]["roi_polygon"]
            if multi_view and measurement_manifest is not None
            else None
        )
        track_bboxes = (
            _load_track_bboxes(tracks_path)
            if multi_view and tracks_path is not None
            else {}
        )

        source_sha256 = _sha256(source)
        records: list[dict[str, Any]] = []
        # views_by_frame[frame_index] -> list of (record, artifact_key, path, bbox_px)
        # artifact_key is "keyframe", "roi_crop", or "event_crop".
        views_by_frame: dict[
            int, list[tuple[dict[str, Any], str, Path, tuple[int, int, int, int] | None]]
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
                views_by_frame[frame_index].append(
                    (record, "keyframe", keyframe_path, None)
                )
                if multi_view:
                    # event_crop takes priority as the "closest view": a
                    # tight, padded crop of the event's own track. Only
                    # when the detector has no bbox to offer here (no
                    # track_id, or the track has no row at this exact
                    # frame) do we fall back to roi_crop as the next-best
                    # available view -- never both, to avoid two nearly-
                    # identical wide shots per event.
                    track_id = event.get("track_id")
                    track_bbox = (
                        track_bboxes.get((frame_index, track_id))
                        if track_id is not None
                        else None
                    )
                    if track_bbox is not None:
                        event_crop_path = frames_dir / f"{event_id}_event_crop.jpg"
                        views_by_frame[frame_index].append(
                            (record, "event_crop", event_crop_path, track_bbox)
                        )
                    elif roi_polygon is not None:
                        roi_crop_path = frames_dir / f"{event_id}_roi_crop.jpg"
                        views_by_frame[frame_index].append(
                            (record, "roi_crop", roi_crop_path, None)
                        )
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

                    for record, artifact_key, path, bbox_px in views_by_frame.get(
                        frame_index, ()
                    ):
                        if artifact_key == "roi_crop":
                            resolved_bbox = _roi_bbox_px(roi_polygon, width, height)
                        elif artifact_key == "event_crop":
                            x1, y1, x2, y2 = bbox_px
                            resolved_bbox = _padded_bbox_px(
                                x1, y1, x2, y2,
                                padding_ratio=self.config.event_crop_padding_ratio,
                                width=width,
                                height=height,
                            )
                        else:
                            resolved_bbox = None
                        artifact = self._write_image_artifact(
                            frame, frame_index, path, bbox_px=resolved_bbox
                        )
                        artifact["path"] = path.relative_to(run_dir).as_posix()
                        record[artifact_key] = artifact

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
            "multi_view_enabled": multi_view,
            "source_video_sha256": source_sha256,
            "selected_events": len(records),
            "keyframes_written": sum("keyframe" in record for record in records),
            "roi_crops_written": sum("roi_crop" in record for record in records),
            "event_crops_written": sum("event_crop" in record for record in records),
            "clips_written": len(clip_specs),
            "manifest": manifest_path.name,
        }
