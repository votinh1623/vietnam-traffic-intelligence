"""Stable records exchanged by perception, analytics, and output adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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


TRACK_CSV_FIELDS = tuple(TrackObservation.__dataclass_fields__)
