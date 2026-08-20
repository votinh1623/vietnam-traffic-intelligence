"""Congestion state machine with explicit temporal hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AnalyticsConfig


STATES = ("NORMAL", "DENSE", "CONGESTED")


@dataclass(frozen=True)
class StateTransition:
    previous: str
    current: str
    timestamp_s: float


class CongestionStateMachine:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.state = "NORMAL"
        self._candidate = self.state
        self._candidate_since = 0.0

    def _target(
        self,
        bbox_union_occupancy: float,
        count: int,
        mean_speed: float | None,
        stalled_dense_fraction: float | None = None,
    ) -> str:
        speed = float("inf") if mean_speed is None else mean_speed
        # fixed_camera keeps the original design: a high count alone must be
        # corroborated by ROI occupancy, because count on a small ground-
        # anchored ROI is noisy. uav_motion drops that co-requirement: with
        # analytics_mode=uav_motion the "ROI" defaults to the full frame, so
        # occupancy is diluted by background/road far from the camera and
        # structurally cannot reach the ground-camera-calibrated thresholds
        # even in a genuinely dense scene (see
        # experiments/uav_pipeline_e2e_v1_20260818 and its Option A rerun).
        # Whole-frame track count is the more trustworthy signal there.
        uav_motion = self.config.analytics_mode == "uav_motion"
        # Detection-independent corroborating signal (see
        # src/vn_traffic/analytics/stillness.py): bbox_union_occupancy/count
        # both depend on the detector resolving individual boxes, which
        # collapses under severe occlusion exactly when congestion is worst.
        # Deliberately NOT gated by `speed <=` below: that speed is the mean
        # of DETECTED roi tracks, which is exactly the signal a severely
        # occluded jam starves -- the stillness fraction already encodes
        # "not moving" directly from pixels, so requiring the detector's own
        # (likely-diluted-by-the-flowing-lane) speed on top would reintroduce
        # the same blind spot this signal exists to avoid.
        stillness_congested = self.config.stillness_enabled and (
            stalled_dense_fraction is not None
            and stalled_dense_fraction
            >= self.config.stillness_congested_enter_fraction
        )
        stillness_remains_congested = self.config.stillness_enabled and (
            stalled_dense_fraction is not None
            and stalled_dense_fraction
            >= self.config.stillness_congested_exit_fraction
        )
        if self.state == "CONGESTED":
            remains_congested = (
                bbox_union_occupancy
                >= self.config.congested_exit_bbox_union_occupancy
                or (
                    count >= self.config.congested_exit_count
                    and (
                        uav_motion
                        or bbox_union_occupancy
                        >= self.config.dense_exit_bbox_union_occupancy
                    )
                    and speed <= self.config.congested_release_speed_px_s
                )
            ) and speed <= self.config.congested_release_speed_px_s
            if remains_congested or stillness_remains_congested:
                return "CONGESTED"
        congested = (
            bbox_union_occupancy
            >= self.config.congested_enter_bbox_union_occupancy
            or (
                count >= self.config.congested_enter_count
                and (
                    uav_motion
                    or bbox_union_occupancy
                    >= self.config.dense_enter_bbox_union_occupancy
                )
                and speed <= self.config.congested_max_speed_px_s
            )
        ) and speed <= self.config.congested_max_speed_px_s
        if congested or stillness_congested:
            return "CONGESTED"

        if self.state in ("DENSE", "CONGESTED"):
            dense = (
                bbox_union_occupancy >= self.config.dense_exit_bbox_union_occupancy
                or (
                    count >= self.config.dense_exit_count
                    and (
                        uav_motion
                        or bbox_union_occupancy
                        >= self.config.dense_exit_bbox_union_occupancy
                    )
                    and speed <= self.config.congested_max_speed_px_s
                )
            )
        else:
            dense = (
                bbox_union_occupancy >= self.config.dense_enter_bbox_union_occupancy
                or (
                    count >= self.config.dense_enter_count
                    and (
                        uav_motion
                        or bbox_union_occupancy
                        >= self.config.dense_exit_bbox_union_occupancy
                    )
                    and speed <= self.config.congested_max_speed_px_s
                )
            )
        return "DENSE" if dense else "NORMAL"

    def update(
        self,
        *,
        timestamp_s: float,
        bbox_union_occupancy: float,
        count: int,
        mean_speed_px_s: float | None,
        stalled_dense_fraction: float | None = None,
    ) -> StateTransition | None:
        target = self._target(
            bbox_union_occupancy, count, mean_speed_px_s, stalled_dense_fraction
        )
        if target == self.state:
            self._candidate = self.state
            self._candidate_since = timestamp_s
            return None
        if target != self._candidate:
            self._candidate = target
            self._candidate_since = timestamp_s
            return None

        current_rank = STATES.index(self.state)
        target_rank = STATES.index(target)
        required_duration = (
            self.config.transition_confirm_s
            if target_rank > current_rank
            else self.config.release_confirm_s
        )
        if timestamp_s - self._candidate_since < required_duration:
            return None
        previous = self.state
        self.state = target
        self._candidate = target
        self._candidate_since = timestamp_s
        return StateTransition(previous, self.state, timestamp_s)
