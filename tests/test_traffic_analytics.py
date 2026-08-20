from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


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


def observation(
    track_id: int, y: float, x: float = 50.0, class_name: str = "car"
) -> TrackObservation:
    return TrackObservation(
        frame_index=0,
        timestamp_s=0.0,
        track_id=track_id,
        class_id=1,
        class_name=class_name,
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

    def test_excludes_non_vehicle_classes_from_all_analytics(self) -> None:
        engine = TrafficAnalytics(self.config)
        first = observation(9, 70.0, class_name="pedestrian")
        second = observation(9, 40.0, class_name="pedestrian")
        self.process(engine, 0, first)
        batch = self.process(engine, 1, second)
        self.assertEqual(batch.events, ())
        self.assertEqual(batch.snapshot.roi_track_count, 0)
        self.assertEqual(batch.snapshot.current_counts, {})
        self.assertEqual(engine.summary()["unique_track_ids"], 0)

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

    def test_uav_motion_mode_allows_count_alone_to_signal_congestion(self) -> None:
        config = replace(
            self.config,
            analytics_mode="uav_motion",
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
        )
        # Same occupancy/count pair that stays NORMAL for fixed_camera in
        # test_calibrated_normal_and_jam_signals_separate: low whole-frame
        # occupancy (diluted by background far from a moving/zooming UAV
        # camera) must not suppress a genuinely high track count.
        uav = CongestionStateMachine(config)
        high_count_low_coverage = dict(
            bbox_union_occupancy=0.13,
            count=195,
            mean_speed_px_s=37.0,
        )
        self.assertIsNone(uav.update(timestamp_s=0.0, **high_count_low_coverage))
        transition = uav.update(timestamp_s=1.0, **high_count_low_coverage)
        self.assertEqual(
            (transition.previous, transition.current),
            ("NORMAL", "CONGESTED"),
        )

    def test_fixed_camera_mode_still_requires_occupancy_with_count(self) -> None:
        config = replace(
            self.config,
            analytics_mode="fixed_camera",
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
        )
        fixed = CongestionStateMachine(config)
        high_count_low_coverage = dict(
            bbox_union_occupancy=0.13,
            count=195,
            mean_speed_px_s=37.0,
        )
        self.assertIsNone(fixed.update(timestamp_s=0.0, **high_count_low_coverage))
        self.assertIsNone(fixed.update(timestamp_s=1.0, **high_count_low_coverage))
        self.assertEqual(fixed.state, "NORMAL")

    def test_stillness_signal_corroborates_congested_despite_low_occupancy_and_count(
        self,
    ) -> None:
        # The real failure case this exists for: a severely occluded jam
        # gives the detector almost nothing to report (low occupancy, low
        # count), but the pixel-level stillness signal is high.
        config = replace(
            self.config,
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
            stillness_enabled=True,
            stillness_congested_enter_fraction=0.30,
            stillness_congested_exit_fraction=0.20,
        )
        machine = CongestionStateMachine(config)
        occluded_jam = dict(
            bbox_union_occupancy=0.05,
            count=2,
            mean_speed_px_s=120.0,
            stalled_dense_fraction=0.40,
        )
        self.assertIsNone(machine.update(timestamp_s=0.0, **occluded_jam))
        transition = machine.update(timestamp_s=1.0, **occluded_jam)
        self.assertEqual(
            (transition.previous, transition.current), ("NORMAL", "CONGESTED")
        )

    def test_stillness_signal_ignored_when_disabled(self) -> None:
        config = replace(
            self.config,
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
            stillness_enabled=False,
        )
        machine = CongestionStateMachine(config)
        occluded_jam = dict(
            bbox_union_occupancy=0.05,
            count=2,
            mean_speed_px_s=120.0,
            stalled_dense_fraction=0.99,
        )
        self.assertIsNone(machine.update(timestamp_s=0.0, **occluded_jam))
        self.assertIsNone(machine.update(timestamp_s=1.0, **occluded_jam))
        self.assertEqual(machine.state, "NORMAL")

    def test_stillness_remains_congested_ignores_detected_speed(self) -> None:
        # Deliberate design choice: once CONGESTED via stillness, a high
        # *detected* mean speed (from the few flowing tracks the detector
        # can see) must not release the state while the stillness fraction
        # itself is still above its own exit threshold -- that detected
        # speed is exactly the signal a severely occluded jam starves.
        config = replace(
            self.config,
            transition_confirm_s=1.0,
            release_confirm_s=1.0,
            stillness_enabled=True,
            stillness_congested_enter_fraction=0.30,
            stillness_congested_exit_fraction=0.20,
        )
        machine = CongestionStateMachine(config)
        occluded_jam = dict(
            bbox_union_occupancy=0.05,
            count=2,
            mean_speed_px_s=120.0,
            stalled_dense_fraction=0.40,
        )
        machine.update(timestamp_s=0.0, **occluded_jam)
        machine.update(timestamp_s=1.0, **occluded_jam)
        self.assertEqual(machine.state, "CONGESTED")

        still_dense_but_fast_detected_speed = dict(
            bbox_union_occupancy=0.05,
            count=2,
            mean_speed_px_s=300.0,
            stalled_dense_fraction=0.25,
        )
        self.assertIsNone(
            machine.update(timestamp_s=2.0, **still_dense_but_fast_detected_speed)
        )
        self.assertEqual(machine.state, "CONGESTED")

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


def _gmc_textured_frame(width: int = 160, height: int = 120, seed: int = 0) -> np.ndarray:
    # Same parameters verified in test_motion.py to give ECC a reliable
    # convergence basin; duplicated here rather than imported so this file
    # stays a self-contained fixture like the rest of the test suite.
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def _gmc_shift(frame: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    height, width = frame.shape[:2]
    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT101
    )


class GmcAnalyticsIntegrationTests(unittest.TestCase):
    def test_warped_roi_keeps_a_stationary_vehicle_inside_the_region(self) -> None:
        # A small ROI (not full-frame) around the image center, so this only
        # passes if the ROI genuinely follows the camera instead of falling
        # back to whole-frame coverage.
        config = AnalyticsConfig(
            analytics_mode="uav_motion",
            gmc_enabled=True,
            gmc_downscale=1,
            roi_polygon=((0.45, 0.4), (0.55, 0.4), (0.55, 0.6), (0.45, 0.6)),
            counting_line=((0.0, 0.5), (1.0, 0.5)),
            transition_confirm_s=100.0,
            release_confirm_s=100.0,
        )
        engine = TrafficAnalytics(config)
        frame0 = _gmc_textured_frame(seed=0)
        shift_x, shift_y = 8.0, -5.0
        frame1 = _gmc_shift(frame0, shift_x, shift_y)

        # Stationary real-world vehicle at the frame-0 ROI center (80, 60).
        first = engine.process(
            frame_index=0,
            timestamp_s=0.0,
            tracks=(observation(1, y=60.0, x=80.0),),
            frame_width=160,
            frame_height=120,
            frame=frame0,
        )
        self.assertEqual(first.snapshot.roi_track_count, 1)

        # The camera panned by (shift_x, shift_y): the same still-parked
        # vehicle's pixel position drifts by the same amount, exactly as a
        # detector would report it in the raw shifted video.
        second = engine.process(
            frame_index=1,
            timestamp_s=1.0,
            tracks=(observation(1, y=60.0 + shift_y, x=80.0 + shift_x),),
            frame_width=160,
            frame_height=120,
            frame=frame1,
        )
        self.assertEqual(second.snapshot.roi_track_count, 1)
        # The warped ROI center should itself have moved by ~(shift_x, shift_y)
        # from its frame-0 position, not stayed put.
        roi_x = [x for x, _ in second.snapshot.roi_polygon_px]
        roi_y = [y for _, y in second.snapshot.roi_polygon_px]
        self.assertAlmostEqual(sum(roi_x) / len(roi_x), 80.0 + shift_x, delta=3.0)
        self.assertAlmostEqual(sum(roi_y) / len(roi_y), 60.0 + shift_y, delta=3.0)

    def test_gmc_enabled_without_a_frame_raises(self) -> None:
        config = AnalyticsConfig(analytics_mode="uav_motion", gmc_enabled=True)
        engine = TrafficAnalytics(config)
        with self.assertRaisesRegex(ValueError, "gmc_enabled"):
            engine.process(
                frame_index=0,
                timestamp_s=0.0,
                tracks=(),
                frame_width=160,
                frame_height=120,
            )


class StillnessAnalyticsIntegrationTests(unittest.TestCase):
    def test_static_textured_scene_reaches_congested_with_zero_tracks(self) -> None:
        # The end-to-end version of the real failure case: no detected
        # tracks at all (occupancy=0, count=0), but a static, visually dense
        # scene (a stand-in for a severely occluded, stalled crowd the
        # detector cannot resolve) still reaches CONGESTED via stillness
        # alone.
        config = AnalyticsConfig(
            roi_polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            counting_line=((0.0, 0.5), (1.0, 0.5)),
            transition_confirm_s=1.0,
            release_confirm_s=2.0,
            stillness_enabled=True,
            stillness_downscale=1,
            stillness_cell_px=8,
            stillness_motion_threshold=2.0,
            stillness_texture_threshold=0.5,
            stillness_congested_enter_fraction=0.5,
            stillness_congested_exit_fraction=0.3,
        )
        engine = TrafficAnalytics(config)
        frame = _gmc_textured_frame(width=160, height=120, seed=7)

        first = engine.process(
            frame_index=0,
            timestamp_s=0.0,
            tracks=(),
            frame_width=160,
            frame_height=120,
            frame=frame,
        )
        self.assertEqual(first.snapshot.congestion_state, "NORMAL")
        self.assertEqual(first.snapshot.roi_track_count, 0)
        self.assertEqual(first.snapshot.bbox_union_occupancy, 0.0)

        engine.process(
            frame_index=1,
            timestamp_s=1.0,
            tracks=(),
            frame_width=160,
            frame_height=120,
            frame=frame,
        )
        third = engine.process(
            frame_index=2,
            timestamp_s=2.0,
            tracks=(),
            frame_width=160,
            frame_height=120,
            frame=frame,
        )
        self.assertEqual(third.snapshot.congestion_state, "CONGESTED")
        self.assertEqual(third.snapshot.roi_track_count, 0)

    def test_stillness_enabled_without_a_frame_raises(self) -> None:
        config = AnalyticsConfig(stillness_enabled=True)
        engine = TrafficAnalytics(config)
        with self.assertRaisesRegex(ValueError, "stillness_enabled"):
            engine.process(
                frame_index=0,
                timestamp_s=0.0,
                tracks=(),
                frame_width=160,
                frame_height=120,
            )


if __name__ == "__main__":
    unittest.main()
