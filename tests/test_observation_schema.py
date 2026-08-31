from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.observation import (  # noqa: E402
    ObservationSchemaError,
    validate_traffic_observation,
)


def video_observation(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "source": "video",
        "run_id": "run73",
        "timestamp_s": 12.5,
        "frame_or_step": 375,
        "position_unit": "normalized_image",
        "speed_unit": "px_s",
        "perception_status": {
            "state": "reliable",
            "reason_codes": [],
            "confidence": None,
        },
        "objects": [
            {
                "object_id": "42",
                "class_name": "motorcycle",
                "region_id": "approach_a",
                "position": [0.41, 0.63],
                "speed": 4.2,
                "confidence": 0.78,
                "observation_kind": "tracked",
            }
        ],
    }
    base.update(overrides)
    return base


def sumo_observation(**overrides) -> dict:
    base = video_observation(
        source="sumo_clean",
        position_unit="lane_road",
        speed_unit="m_s",
    )
    base["objects"][0] = {
        **base["objects"][0],
        "object_id": "veh_12",
        "observation_kind": "simulator_ground_truth",
    }
    base.update(overrides)
    return base


class TrafficObservationSchemaTests(unittest.TestCase):
    def test_accepts_a_valid_video_observation(self) -> None:
        validate_traffic_observation(video_observation())

    def test_accepts_a_valid_sumo_clean_observation(self) -> None:
        validate_traffic_observation(sumo_observation())

    def test_rejects_video_source_with_sumo_units(self) -> None:
        payload = video_observation(position_unit="lane_road", speed_unit="m_s")
        with self.assertRaisesRegex(ObservationSchemaError, "source=video requires"):
            validate_traffic_observation(payload)

    def test_rejects_sumo_clean_source_with_video_units(self) -> None:
        payload = sumo_observation(
            position_unit="normalized_image", speed_unit="px_s"
        )
        with self.assertRaisesRegex(
            ObservationSchemaError, "source=sumo_clean/sumo_noisy requires"
        ):
            validate_traffic_observation(payload)

    def test_rejects_video_object_claiming_simulator_ground_truth(self) -> None:
        payload = video_observation()
        payload["objects"][0]["observation_kind"] = "simulator_ground_truth"
        with self.assertRaisesRegex(
            ObservationSchemaError, "'tracked' for source=video"
        ):
            validate_traffic_observation(payload)

    def test_rejects_sumo_clean_object_claiming_tracked(self) -> None:
        payload = sumo_observation()
        payload["objects"][0]["observation_kind"] = "tracked"
        with self.assertRaisesRegex(
            ObservationSchemaError, "'simulator_ground_truth' for source=sumo_clean"
        ):
            validate_traffic_observation(payload)

    def test_rejects_duplicate_object_ids(self) -> None:
        payload = video_observation()
        payload["objects"] = payload["objects"] * 2
        with self.assertRaisesRegex(ObservationSchemaError, "duplicate object_id"):
            validate_traffic_observation(payload)

    def test_rejects_unknown_perception_status_state(self) -> None:
        payload = video_observation()
        payload["perception_status"]["state"] = "definitely_fine"
        with self.assertRaisesRegex(
            ObservationSchemaError, "perception_status.state must be one of"
        ):
            validate_traffic_observation(payload)

    def test_rejects_out_of_range_confidence(self) -> None:
        payload = video_observation()
        payload["objects"][0]["confidence"] = 1.5
        with self.assertRaisesRegex(ObservationSchemaError, "confidence"):
            validate_traffic_observation(payload)

    def test_allows_null_confidence_and_speed(self) -> None:
        payload = video_observation()
        payload["objects"][0]["confidence"] = None
        payload["objects"][0]["speed"] = None
        validate_traffic_observation(payload)

    def test_rejects_negative_speed(self) -> None:
        payload = video_observation()
        payload["objects"][0]["speed"] = -1.0
        with self.assertRaisesRegex(ObservationSchemaError, "speed cannot be negative"):
            validate_traffic_observation(payload)

    def test_rejects_extra_top_level_field(self) -> None:
        payload = video_observation()
        payload["extra_field"] = True
        with self.assertRaisesRegex(ObservationSchemaError, "unsupported fields"):
            validate_traffic_observation(payload)


if __name__ == "__main__":
    unittest.main()
