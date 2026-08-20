"""Deterministic trajectory, counting, occupancy, and congestion analytics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from ..config import AnalyticsConfig
from ..schemas import (
    ANALYTICS_SCHEMA_VERSION,
    AnalyticsBatch,
    AnalyticsSnapshot,
    TrackObservation,
)
from .geometry import (
    Point,
    point_in_polygon,
    segments_intersect,
    stable_line_side,
    to_pixels,
)
from .motion import GlobalMotionCompensator
from .occupancy import BBoxUnionOccupancy
from .state import CongestionStateMachine
from .stillness import StillnessTracker


@dataclass
class TrackMemory:
    last_point: Point
    last_timestamp_s: float
    stable_side: int | None = None
    stable_point: Point | None = None
    counted_directions: set[str] = field(default_factory=set)
    trajectory: deque[Point] = field(default_factory=deque)
    stop_candidate_since_s: float | None = None
    prolonged_stop_active: bool = False


class TrafficAnalytics:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.state_machine = CongestionStateMachine(config)
        self._tracks: dict[int, TrackMemory] = {}
        self._seen_track_ids: set[int] = set()
        self._crossings: dict[str, Counter[str]] = {
            "up": Counter(),
            "down": Counter(),
        }
        self._event_number = 0
        self._frames_by_state: Counter[str] = Counter()
        self._transition_count = 0
        self._prolonged_stop_count = 0
        self._max_bbox_union_occupancy = 0.0
        self._max_roi_track_count = 0
        self._occupancy = BBoxUnionOccupancy(
            config.roi_polygon,
            config.occupancy_grid_size_px,
        )
        self._motion = (
            GlobalMotionCompensator(downscale=config.gmc_downscale)
            if config.gmc_enabled
            else None
        )
        self._stillness = (
            StillnessTracker(
                downscale=config.stillness_downscale,
                cell_px=config.stillness_cell_px,
                motion_threshold=config.stillness_motion_threshold,
                texture_threshold=config.stillness_texture_threshold,
            )
            if config.stillness_enabled
            else None
        )

    def _event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self._event_number += 1
        return {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "event_id": f"event-{self._event_number:06d}",
            "event_type": event_type,
            **payload,
        }

    def process(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        tracks: tuple[TrackObservation, ...],
        frame_width: int,
        frame_height: int,
        frame: Any = None,
    ) -> AnalyticsBatch:
        # roi/line here are always the reference-frame (frame 0) pixel
        # positions. Without GMC that reference frame IS the current frame,
        # so this is already the effective geometry. With GMC, warp_points()
        # below re-projects them into the current frame instead.
        roi = to_pixels(self.config.roi_polygon, frame_width, frame_height)
        line_start, line_end = to_pixels(
            self.config.counting_line, frame_width, frame_height
        )
        if self._motion is not None:
            if frame is None:
                raise ValueError(
                    "analytics.gmc_enabled requires the raw frame to be passed "
                    "to TrafficAnalytics.process(frame=...)"
                )
            self._motion.update(frame)
            roi = tuple(self._motion.warp_points(roi))
            line_start, line_end = self._motion.warp_points([line_start, line_end])
        if self._stillness is not None and frame is None:
            raise ValueError(
                "analytics.stillness_enabled requires the raw frame to be passed "
                "to TrafficAnalytics.process(frame=...)"
            )
        boxes: list[tuple[float, float, float, float]] = []
        current_counts: Counter[str] = Counter()
        current_speeds: list[float] = []
        roi_track_ids: set[int] = set()
        events: list[dict[str, Any]] = []

        for track in tracks:
            if track.class_name not in self.config.included_classes:
                continue
            point = ((track.x1 + track.x2) / 2.0, (track.y1 + track.y2) / 2.0)
            inside_roi = point_in_polygon(point, roi)
            boxes.append((track.x1, track.y1, track.x2, track.y2))
            if inside_roi:
                current_counts[track.class_name] += 1
                if track.track_id is not None:
                    roi_track_ids.add(track.track_id)

            if track.track_id is None:
                continue
            self._seen_track_ids.add(track.track_id)
            memory = self._tracks.get(track.track_id)
            if memory is None:
                side = stable_line_side(
                    point,
                    line_start,
                    line_end,
                    self.config.line_tolerance_px,
                )
                memory = TrackMemory(
                    last_point=point,
                    last_timestamp_s=timestamp_s,
                    stable_side=side if side else None,
                    stable_point=point if side else None,
                    trajectory=deque([point], maxlen=self.config.trajectory_history),
                )
                self._tracks[track.track_id] = memory
                continue

            elapsed = timestamp_s - memory.last_timestamp_s
            speed = None
            if elapsed > 0:
                speed = math.dist(memory.last_point, point) / elapsed
                if inside_roi:
                    current_speeds.append(speed)

            stop_eligible = (
                self.config.prolonged_stop_enabled
                and inside_roi
                and track.class_name in self.config.prolonged_stop_classes
                and speed is not None
                and 0 < elapsed <= self.config.prolonged_stop_max_gap_s
            )
            if not stop_eligible:
                memory.stop_candidate_since_s = None
                memory.prolonged_stop_active = False
            elif speed <= self.config.prolonged_stop_max_speed_px_s:
                if memory.stop_candidate_since_s is None:
                    memory.stop_candidate_since_s = memory.last_timestamp_s
                stopped_duration_s = timestamp_s - memory.stop_candidate_since_s
                if (
                    not memory.prolonged_stop_active
                    and stopped_duration_s >= self.config.prolonged_stop_min_duration_s
                ):
                    memory.prolonged_stop_active = True
                    self._prolonged_stop_count += 1
                    events.append(
                        self._event(
                            "prolonged_stop",
                            timestamp_s=timestamp_s,
                            frame_index=frame_index,
                            track_id=track.track_id,
                            class_id=track.class_id,
                            class_name=track.class_name,
                            measurements={
                                "speed_px_s": speed,
                                "stopped_duration_s": stopped_duration_s,
                            },
                        )
                    )
            elif speed >= self.config.prolonged_stop_release_speed_px_s:
                memory.stop_candidate_since_s = None
                memory.prolonged_stop_active = False
            elif not memory.prolonged_stop_active:
                # Entry requires continuous evidence below the entry threshold;
                # the hysteresis band only preserves an already-active alert.
                memory.stop_candidate_since_s = None

            current_side = stable_line_side(
                point,
                line_start,
                line_end,
                self.config.line_tolerance_px,
            )
            if (
                inside_roi
                and current_side
                and memory.stable_side is not None
                and current_side != memory.stable_side
                and memory.stable_point is not None
                and segments_intersect(
                    memory.stable_point, point, line_start, line_end
                )
            ):
                direction = "up" if point[1] < memory.stable_point[1] else "down"
                if direction not in memory.counted_directions:
                    memory.counted_directions.add(direction)
                    self._crossings[direction][track.class_name] += 1
                    events.append(
                        self._event(
                            "line_crossing",
                            timestamp_s=timestamp_s,
                            frame_index=frame_index,
                            track_id=track.track_id,
                            class_id=track.class_id,
                            class_name=track.class_name,
                            direction=direction,
                            measurements={"speed_px_s": speed},
                        )
                    )
            if current_side:
                memory.stable_side = current_side
                memory.stable_point = point
            memory.last_point = point
            memory.last_timestamp_s = timestamp_s
            memory.trajectory.append(point)

        bbox_union_occupancy = self._occupancy.measure(
            boxes,
            frame_width,
            frame_height,
            roi_polygon_px=roi if self._motion is not None else None,
        )
        mean_speed = (
            sum(current_speeds) / len(current_speeds) if current_speeds else None
        )
        stalled_dense_fraction = (
            self._stillness.update(frame, roi_polygon_px=roi)
            if self._stillness is not None
            else None
        )
        transition = self.state_machine.update(
            timestamp_s=timestamp_s,
            bbox_union_occupancy=bbox_union_occupancy,
            count=len(roi_track_ids),
            mean_speed_px_s=mean_speed,
            stalled_dense_fraction=stalled_dense_fraction,
        )
        if transition is not None:
            self._transition_count += 1
            events.append(
                self._event(
                    "congestion_transition",
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                    previous_state=transition.previous,
                    current_state=transition.current,
                    measurements={
                        "bbox_union_occupancy": bbox_union_occupancy,
                        "roi_track_count": len(roi_track_ids),
                        "mean_speed_px_s": mean_speed,
                        "stalled_dense_fraction": stalled_dense_fraction,
                    },
                )
            )

        state = self.state_machine.state
        self._frames_by_state[state] += 1
        self._max_bbox_union_occupancy = max(
            self._max_bbox_union_occupancy,
            bbox_union_occupancy,
        )
        self._max_roi_track_count = max(
            self._max_roi_track_count, len(roi_track_ids)
        )
        snapshot = AnalyticsSnapshot(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            congestion_state=state,
            roi_track_count=len(roi_track_ids),
            bbox_union_occupancy=bbox_union_occupancy,
            mean_speed_px_s=mean_speed,
            current_counts=dict(sorted(current_counts.items())),
            cumulative_crossings={
                direction: dict(sorted(counts.items()))
                for direction, counts in self._crossings.items()
            },
            roi_polygon_px=tuple(roi),
            counting_line_px=(line_start, line_end),
            stalled_dense_fraction=stalled_dense_fraction,
        )
        return AnalyticsBatch(snapshot=snapshot, events=tuple(events))

    def summary(self) -> dict[str, Any]:
        if self.config.gmc_enabled:
            mode_note = (
                "analytics_mode=uav_motion, gmc_enabled=true: the roi_polygon/"
                "counting_line defined on frame 0 are re-projected into every "
                "frame using ECC-estimated camera motion (see "
                "src/vn_traffic/analytics/motion.py), instead of collapsing to "
                "the full frame. This is 2D image-plane motion compensation "
                "only (no GPS/BEV georeferencing) and can lose lock under a "
                "hard scene cut, fast motion, or low-texture frames -- see "
                "gmc_consecutive_failures_at_end below."
            )
        elif self.config.analytics_mode == "uav_motion":
            mode_note = (
                "analytics_mode=uav_motion: the analytics region defaults to the "
                "full frame instead of a hand-drawn ROI, because a fixed "
                "ground-anchored polygon misplaces the congestion region within "
                "seconds under camera pan/zoom (see "
                "experiments/uav_pipeline_e2e_v1_20260818). This is not camera-"
                "motion compensation: occupancy still shifts with zoom, and count "
                "is a whole-frame track count rather than a location-specific one."
            )
        else:
            mode_note = (
                "analytics_mode=fixed_camera: ROI/occupancy assume a static "
                "camera and are not valid if the source pans or zooms."
            )
        return {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "analytics_enabled": True,
            "analytics_mode": self.config.analytics_mode,
            "gmc_enabled": self.config.gmc_enabled,
            "gmc_consecutive_failures_at_end": (
                self._motion.consecutive_failures if self._motion is not None else None
            ),
            "stillness_enabled": self.config.stillness_enabled,
            "state_frames": dict(self._frames_by_state),
            "congestion_transitions": self._transition_count,
            "prolonged_stop_events": self._prolonged_stop_count,
            "cumulative_crossings": {
                direction: dict(sorted(counts.items()))
                for direction, counts in self._crossings.items()
            },
            "unique_track_ids": len(self._seen_track_ids),
            "max_roi_track_count": self._max_roi_track_count,
            "max_bbox_union_occupancy": self._max_bbox_union_occupancy,
            "occupancy_grid_size_px": self.config.occupancy_grid_size_px,
            "claim_boundary": (
                "Counts can be biased by ByteTrack ID switches or fragmentation; "
                "VisDrone trajectory count error has been measured, but demo-video "
                "count error has not. Congestion "
                "thresholds have only initial two-video demo calibration and may "
                "not transfer to another camera or viewpoint. Bbox-union "
                "occupancy is image-plane box coverage, not physical road "
                "occupancy; boxes include background and no BEV calibration is "
                "applied. Prolonged-stop alerts use image-plane centroid speed and "
                "can be invalid under camera motion or identity errors. "
                + mode_note
                + (
                    " stillness_enabled=true: CONGESTED can also be reached/held "
                    "via a detection-independent optical-flow+texture signal "
                    "(src/vn_traffic/analytics/stillness.py), not gated by "
                    "detected mean speed, specifically to corroborate severe, "
                    "heavily-occluded jams the detector's own boxes may miss. "
                    "Its thresholds are demo-calibrated on one real frame, not "
                    "validated across multiple scenes -- see benchmark "
                    "protocol."
                    if self.config.stillness_enabled
                    else ""
                )
            ),
        }
