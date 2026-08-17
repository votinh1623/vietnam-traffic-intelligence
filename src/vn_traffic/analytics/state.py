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

    def _target(self, occupancy: float, count: int, mean_speed: float | None) -> str:
        speed = float("inf") if mean_speed is None else mean_speed
        if self.state == "CONGESTED":
            remains_congested = (
                occupancy >= self.config.congested_exit_occupancy
                or (
                    count >= self.config.congested_exit_count
                    and speed <= self.config.congested_release_speed_px_s
                )
            ) and speed <= self.config.congested_release_speed_px_s
            if remains_congested:
                return "CONGESTED"
        congested = (
            occupancy >= self.config.congested_enter_occupancy
            or (
                count >= self.config.congested_enter_count
                and speed <= self.config.congested_max_speed_px_s
            )
        ) and speed <= self.config.congested_max_speed_px_s
        if congested:
            return "CONGESTED"

        if self.state in ("DENSE", "CONGESTED"):
            dense = (
                occupancy >= self.config.dense_exit_occupancy
                or (
                    count >= self.config.dense_exit_count
                    and speed <= self.config.congested_max_speed_px_s
                )
            )
        else:
            dense = (
                occupancy >= self.config.dense_enter_occupancy
                or (
                    count >= self.config.dense_enter_count
                    and speed <= self.config.congested_max_speed_px_s
                )
            )
        return "DENSE" if dense else "NORMAL"

    def update(
        self,
        *,
        timestamp_s: float,
        occupancy: float,
        count: int,
        mean_speed_px_s: float | None,
    ) -> StateTransition | None:
        target = self._target(occupancy, count, mean_speed_px_s)
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
