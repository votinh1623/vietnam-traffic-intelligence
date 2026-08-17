from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.analytics.engine import TrafficAnalytics  # noqa: E402
from vn_traffic.analytics.geometry import (  # noqa: E402
    point_in_polygon,
    polygon_rectangle_intersection_area,
)
from vn_traffic.analytics.state import CongestionStateMachine  # noqa: E402
from vn_traffic.config import AnalyticsConfig  # noqa: E402
from vn_traffic.schemas import TrackObservation  # noqa: E402


def observation(track_id: int, y: float, x: float = 50.0) -> TrackObservation:
    return TrackObservation(
        frame_index=0,
        timestamp_s=0.0,
        track_id=track_id,
        class_id=1,
        class_name="car",
        confidence=0.9,
        x1=x - 5,
        y1=y - 5,
        x2=x + 5,
        y2=y + 5,
    )


class GeometryTests(unittest.TestCase):
    def test_polygon_membership_and_rectangle_intersection(self) -> None:
        square = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
        self.assertTrue(point_in_polygon((50.0, 50.0), square))
        self.assertFalse(point_in_polygon((150.0, 50.0), square))
        area = polygon_rectangle_intersection_area(square, (90.0, 90.0, 110.0, 110.0))
        self.assertAlmostEqual(area, 100.0)


class TrafficAnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AnalyticsConfig(
            roi_polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            counting_line=((0.0, 0.5), (1.0, 0.5)),
            line_tolerance_px=0.0,
            transition_confirm_s=100.0,
            release_confirm_s=100.0,
        )

    def process(self, engine: TrafficAnalytics, frame: int, track: TrackObservation):
        return engine.process(
            frame_index=frame,
            timestamp_s=float(frame),
            tracks=(track,),
            frame_width=100,
            frame_height=100,
        )

    def test_counts_each_direction_once_per_track(self) -> None:
        engine = TrafficAnalytics(self.config)
        self.process(engine, 0, observation(7, 70.0))
        upward = self.process(engine, 1, observation(7, 40.0))
        downward = self.process(engine, 2, observation(7, 70.0))
        repeated_upward = self.process(engine, 3, observation(7, 40.0))

        self.assertEqual(upward.events[0]["direction"], "up")
        self.assertEqual(downward.events[0]["direction"], "down")
        self.assertEqual(repeated_upward.events, ())
        self.assertEqual(
            repeated_upward.snapshot.cumulative_crossings,
            {"up": {"car": 1}, "down": {"car": 1}},
        )

    def test_roi_filters_current_counts(self) -> None:
        config = replace(
            self.config,
            roi_polygon=((0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)),
        )
        engine = TrafficAnalytics(config)
        batch = self.process(engine, 0, observation(1, 20.0, x=80.0))
        self.assertEqual(batch.snapshot.roi_track_count, 0)
        self.assertEqual(batch.snapshot.current_counts, {})

    def test_congestion_hysteresis_requires_time_confirmation(self) -> None:
        config = replace(
            self.config,
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
        )
        machine = CongestionStateMachine(config)
        high = dict(occupancy=0.7, count=60, mean_speed_px_s=5.0)
        low = dict(occupancy=0.0, count=0, mean_speed_px_s=100.0)

        self.assertIsNone(machine.update(timestamp_s=0.0, **high))
        self.assertIsNone(machine.update(timestamp_s=0.5, **high))
        transition = machine.update(timestamp_s=1.0, **high)
        self.assertEqual((transition.previous, transition.current), ("NORMAL", "CONGESTED"))
        self.assertIsNone(machine.update(timestamp_s=2.0, **low))
        self.assertIsNone(machine.update(timestamp_s=3.5, **low))
        transition = machine.update(timestamp_s=4.0, **low)
        self.assertEqual((transition.previous, transition.current), ("CONGESTED", "NORMAL"))

    def test_calibrated_normal_and_jam_signals_separate(self) -> None:
        config = replace(
            self.config,
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
        )
        normal = CongestionStateMachine(config)
        moving_traffic = dict(occupancy=0.20, count=39, mean_speed_px_s=100.0)
        self.assertIsNone(normal.update(timestamp_s=0.0, **moving_traffic))
        self.assertIsNone(normal.update(timestamp_s=2.0, **moving_traffic))
        self.assertEqual(normal.state, "NORMAL")

        jam = CongestionStateMachine(config)
        stopped_dense_traffic = dict(
            occupancy=0.65,
            count=23,
            mean_speed_px_s=80.0,
        )
        self.assertIsNone(jam.update(timestamp_s=0.0, **stopped_dense_traffic))
        transition = jam.update(timestamp_s=1.0, **stopped_dense_traffic)
        self.assertEqual((transition.previous, transition.current), ("NORMAL", "CONGESTED"))


if __name__ == "__main__":
    unittest.main()
