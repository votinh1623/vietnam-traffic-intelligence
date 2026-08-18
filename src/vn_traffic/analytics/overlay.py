"""OpenCV rendering kept separate from deterministic analytics logic."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import AnalyticsConfig
from ..schemas import AnalyticsSnapshot


class AnalyticsOverlay:
    COLORS = {
        "NORMAL": (0, 200, 0),
        "DENSE": (0, 165, 255),
        "CONGESTED": (0, 0, 255),
    }

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def draw(self, frame: Any, snapshot: AnalyticsSnapshot) -> Any:
        # Drawn from the snapshot's own (possibly GMC-warped) geometry rather
        # than recomputed from the static config, so the overlay always shows
        # the region analytics actually used for this frame.
        roi = np.array(snapshot.roi_polygon_px, dtype=np.int32)
        line = snapshot.counting_line_px
        color = self.COLORS[snapshot.congestion_state]
        cv2.polylines(frame, [roi], isClosed=True, color=(255, 180, 0), thickness=2)
        cv2.line(
            frame,
            tuple(map(int, line[0])),
            tuple(map(int, line[1])),
            (255, 0, 255),
            2,
        )
        speed = (
            "n/a"
            if snapshot.mean_speed_px_s is None
            else f"{snapshot.mean_speed_px_s:.1f}px/s"
        )
        up_total = sum(snapshot.cumulative_crossings.get("up", {}).values())
        down_total = sum(snapshot.cumulative_crossings.get("down", {}).values())
        lines = (
            f"State: {snapshot.congestion_state}",
            f"ROI tracks: {snapshot.roi_track_count}",
            f"BBox union occupancy: {snapshot.bbox_union_occupancy:.3f}",
            f"Mean speed: {speed}",
            f"Crossings up/down: {up_total}/{down_total}",
        )
        for index, text in enumerate(lines):
            y = 30 + index * 25
            cv2.putText(
                frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame
