"""Stable records exchanged by perception, analytics, and output adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


ANALYTICS_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    timestamp_s: float
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionResult:
    annotated_frame: Any
    tracks: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class AnalyticsSnapshot:
    frame_index: int
    timestamp_s: float
    congestion_state: str
    # Independent of congestion_state: "reliable" or "detection_silence"
    # (the perception stage produced zero raw detections for at least
    # analytics.perception.detection_silence_min_duration_s). A dense/
    # congested state reached during silence via the stillness signal is
    # still trustworthy corroborated evidence; congestion_state itself is
    # never forced to UNKNOWN here. It is the emitted congestion_transition
    # event's current_state -- what actually reaches VLM/LLM reports -- that
    # gets overridden to "UNKNOWN" instead of "NORMAL" during silence, so a
    # silent recall collapse cannot be reported downstream as a clear road.
    perception_status: str
    roi_track_count: int
    bbox_union_occupancy: float
    mean_speed_px_s: float | None
    current_counts: dict[str, int]
    cumulative_crossings: dict[str, dict[str, int]]
    roi_polygon_px: tuple[tuple[float, float], ...]
    counting_line_px: tuple[tuple[float, float], tuple[float, float]]
    stalled_dense_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "congestion_state": self.congestion_state,
            "perception_status": self.perception_status,
            "roi_track_count": self.roi_track_count,
            "bbox_union_occupancy": self.bbox_union_occupancy,
            "mean_speed_px_s": self.mean_speed_px_s,
            "stalled_dense_fraction": self.stalled_dense_fraction,
            "current_counts_json": json.dumps(
                self.current_counts, sort_keys=True, separators=(",", ":")
            ),
            "cumulative_crossings_json": json.dumps(
                self.cumulative_crossings, sort_keys=True, separators=(",", ":")
            ),
        }


@dataclass(frozen=True)
class AnalyticsBatch:
    snapshot: AnalyticsSnapshot | None
    events: tuple[dict[str, Any], ...]


TRACK_CSV_FIELDS = tuple(TrackObservation.__dataclass_fields__)
ANALYTICS_CSV_FIELDS = (
    "frame_index",
    "timestamp_s",
    "congestion_state",
    "perception_status",
    "roi_track_count",
    "bbox_union_occupancy",
    "mean_speed_px_s",
    "stalled_dense_fraction",
    "current_counts_json",
    "cumulative_crossings_json",
)
