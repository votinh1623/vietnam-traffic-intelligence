"""OpenCV rendering kept separate from deterministic analytics logic."""

from __future__ import annotations

from typing import Any

import cv2

from ..config import AnalyticsConfig
from ..schemas import AnalyticsSnapshot


def draw_frame_stats(frame: Any, *, frame_index: int, fps: float | None, vehicle_count: int) -> Any:
    """Always-on demo overlay: frame number, live processing FPS, vehicle count.

    Independent of analytics being enabled -- drawn on every run so the
    output video is reviewable as a standalone demo (frame/fps/count are
    the baseline expectation, not something only a full analytics profile
    should show).
    """
    fps_text = "n/a" if fps is None else f"{fps:.1f}"
    lines = [
        f"Frame: {frame_index}",
        f"FPS: {fps_text}",
        f"Vehicles: {vehicle_count}",
    ]
    # Top-right corner so this never overlaps the analytics overlay, which
    # occupies the top-left when analytics is enabled.
    frame_width = frame.shape[1]
    for index, text in enumerate(lines):
        y = 30 + index * 28
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = frame_width - text_w - 12
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
        )
    return frame


class AnalyticsOverlay:
    COLORS = {
        "NORMAL": (0, 200, 0),
        "DENSE": (0, 165, 255),
        "CONGESTED": (0, 0, 255),
    }

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def draw(self, frame: Any, snapshot: AnalyticsSnapshot) -> Any:
        # Counting line still drawn from the snapshot's own (possibly
        # GMC-warped) geometry, so it always shows where analytics actually
        # measured this frame. ROI polygon itself is intentionally not drawn
        # -- it still applies to the underlying occupancy/count math, just
        # no longer rendered on screen.
        line = snapshot.counting_line_px
        color = self.COLORS[snapshot.congestion_state]
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
        lines = [
            f"State: {snapshot.congestion_state}",
            f"ROI tracks: {snapshot.roi_track_count}",
            f"BBox union occupancy: {snapshot.bbox_union_occupancy:.3f}",
            f"Mean speed: {speed}",
            f"Crossings up/down: {up_total}/{down_total}",
        ]
        if snapshot.stalled_dense_fraction is not None:
            lines.append(f"Stalled-dense fraction: {snapshot.stalled_dense_fraction:.3f}")
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
