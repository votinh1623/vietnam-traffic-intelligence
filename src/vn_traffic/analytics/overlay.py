"""OpenCV rendering kept separate from deterministic analytics logic."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import AnalyticsConfig
from ..schemas import AnalyticsSnapshot
from .geometry import to_pixels


class AnalyticsOverlay:
    COLORS = {
        "NORMAL": (0, 200, 0),
        "DENSE": (0, 165, 255),
        "CONGESTED": (0, 0, 255),
    }

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def draw(self, frame: Any, snapshot: AnalyticsSnapshot) -> Any:
        height, width = frame.shape[:2]
        roi = np.array(
            to_pixels(self.config.roi_polygon, width, height), dtype=np.int32
        )
        line = to_pixels(self.config.counting_line, width, height)
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
            f"Occupancy: {snapshot.occupancy:.3f}",
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
