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
from vn_traffic.analytics.occupancy import BBoxUnionOccupancy  # noqa: E402
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


class BBoxUnionOccupancyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full_roi = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    def test_identical_boxes_are_counted_once(self) -> None:
        metric = BBoxUnionOccupancy(self.full_roi)
        box = (10.0, 10.0, 30.0, 30.0)
        self.assertAlmostEqual(metric.measure((box, box), 100, 100), 0.04)

    def test_partially_overlapping_boxes_use_union_area(self) -> None:
        metric = BBoxUnionOccupancy(self.full_roi)
        boxes = ((10.0, 10.0, 30.0, 30.0), (20.0, 10.0, 40.0, 30.0))
        self.assertAlmostEqual(metric.measure(boxes, 100, 100), 0.06)

    def test_only_coverage_inside_roi_is_counted(self) -> None:
        left_half_roi = ((0.0, 0.0), (0.49, 0.0), (0.49, 1.0), (0.0, 1.0))
        metric = BBoxUnionOccupancy(left_half_roi)
        self.assertAlmostEqual(
            metric.measure(((40.0, 10.0, 60.0, 30.0),), 100, 100),
            0.04,
        )


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
        high = dict(bbox_union_occupancy=0.7, count=60, mean_speed_px_s=5.0)
        low = dict(bbox_union_occupancy=0.0, count=0, mean_speed_px_s=100.0)

        self.assertIsNone(machine.update(timestamp_s=0.0, **high))
        self.assertIsNone(machine.update(timestamp_s=0.5, **high))
        transition = machine.update(timestamp_s=1.0, **high)
        self.assertEqual(
            (transition.previous, transition.current),
            ("NORMAL", "CONGESTED"),
        )
        self.assertIsNone(machine.update(timestamp_s=2.0, **low))
        self.assertIsNone(machine.update(timestamp_s=3.5, **low))
        transition = machine.update(timestamp_s=4.0, **low)
        self.assertEqual(
            (transition.previous, transition.current),
            ("CONGESTED", "NORMAL"),
        )

    def test_calibrated_normal_and_jam_signals_separate(self) -> None:
        config = replace(
            self.config,
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
        )
        normal = CongestionStateMachine(config)
        moving_traffic = dict(
            bbox_union_occupancy=0.20,
            count=39,
            mean_speed_px_s=100.0,
        )
        self.assertIsNone(normal.update(timestamp_s=0.0, **moving_traffic))
        self.assertIsNone(normal.update(timestamp_s=2.0, **moving_traffic))
        self.assertEqual(normal.state, "NORMAL")

        high_count_low_coverage = dict(
            bbox_union_occupancy=0.20,
            count=60,
            mean_speed_px_s=50.0,
        )
        self.assertIsNone(normal.update(timestamp_s=3.0, **high_count_low_coverage))
        self.assertIsNone(normal.update(timestamp_s=5.0, **high_count_low_coverage))
        self.assertEqual(normal.state, "NORMAL")

        jam = CongestionStateMachine(config)
        stopped_dense_traffic = dict(
            bbox_union_occupancy=0.65,
            count=23,
            mean_speed_px_s=80.0,
        )
        self.assertIsNone(jam.update(timestamp_s=0.0, **stopped_dense_traffic))
        transition = jam.update(timestamp_s=1.0, **stopped_dense_traffic)
        self.assertEqual(
            (transition.previous, transition.current),
            ("NORMAL", "CONGESTED"),
        )

    def test_prolonged_stop_alert_has_duration_release_and_no_repeat(self) -> None:
        config = replace(
            self.config,
            prolonged_stop_enabled=True,
            prolonged_stop_classes=("car",),
            prolonged_stop_max_speed_px_s=1.0,
            prolonged_stop_release_speed_px_s=5.0,
            prolonged_stop_min_duration_s=2.0,
            prolonged_stop_max_gap_s=1.0,
        )
        engine = TrafficAnalytics(config)
        self.process(engine, 0, observation(9, 20.0, x=20.0))
        self.assertEqual(self.process(engine, 1, observation(9, 20.0, x=20.0)).events, ())
        alert = self.process(engine, 2, observation(9, 20.0, x=20.0))
        stop_events = [
            event for event in alert.events if event["event_type"] == "prolonged_stop"
        ]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0]["measurements"]["stopped_duration_s"], 2.0)
        repeated = self.process(engine, 3, observation(9, 20.0, x=20.0))
        self.assertFalse(
            any(event["event_type"] == "prolonged_stop" for event in repeated.events)
        )

        self.process(engine, 4, observation(9, 20.0, x=80.0))
        self.process(engine, 5, observation(9, 20.0, x=80.0))
        second_alert = self.process(engine, 6, observation(9, 20.0, x=80.0))
        self.assertTrue(
            any(event["event_type"] == "prolonged_stop" for event in second_alert.events)
        )
        self.assertEqual(engine.summary()["prolonged_stop_events"], 2)

    def test_prolonged_stop_does_not_bridge_long_tracking_gap(self) -> None:
        config = replace(
            self.config,
            prolonged_stop_enabled=True,
            prolonged_stop_classes=("car",),
            prolonged_stop_min_duration_s=2.0,
            prolonged_stop_max_gap_s=1.0,
        )
        engine = TrafficAnalytics(config)
        self.process(engine, 0, observation(4, 20.0))
        batch = self.process(engine, 10, observation(4, 20.0))
        self.assertFalse(
            any(event["event_type"] == "prolonged_stop" for event in batch.events)
        )

    def test_prolonged_stop_entry_requires_continuous_low_speed(self) -> None:
        config = replace(
            self.config,
            prolonged_stop_enabled=True,
            prolonged_stop_classes=("car",),
            prolonged_stop_max_speed_px_s=1.0,
            prolonged_stop_release_speed_px_s=5.0,
            prolonged_stop_min_duration_s=2.0,
            prolonged_stop_max_gap_s=1.0,
        )
        engine = TrafficAnalytics(config)
        self.process(engine, 0, observation(8, 20.0, x=20.0))
        self.process(engine, 1, observation(8, 20.0, x=20.0))
        self.process(engine, 2, observation(8, 20.0, x=23.0))
        self.process(engine, 3, observation(8, 20.0, x=23.0))
        alert = self.process(engine, 4, observation(8, 20.0, x=23.0))
        self.assertTrue(
            any(event["event_type"] == "prolonged_stop" for event in alert.events)
        )


if __name__ == "__main__":
    unittest.main()
