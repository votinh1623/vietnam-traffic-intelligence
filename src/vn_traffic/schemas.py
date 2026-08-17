"""Stable records exchanged by perception, analytics, and output adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


ANALYTICS_SCHEMA_VERSION = 2


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
    roi_track_count: int
    bbox_union_occupancy: float
    mean_speed_px_s: float | None
    current_counts: dict[str, int]
    cumulative_crossings: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "congestion_state": self.congestion_state,
            "roi_track_count": self.roi_track_count,
            "bbox_union_occupancy": self.bbox_union_occupancy,
            "mean_speed_px_s": self.mean_speed_px_s,
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
    "roi_track_count",
    "bbox_union_occupancy",
    "mean_speed_px_s",
    "current_counts_json",
    "cumulative_crossings_json",
)
