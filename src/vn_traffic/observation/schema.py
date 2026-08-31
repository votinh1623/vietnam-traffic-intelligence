"""TrafficObservation schema v1.

The common contract between the real-video pipeline and the SUMO
simulation branch (see reports/ke-hoach-pipeline-va-mo-phong.md section 7).
Locked here, ahead of either VideoObservationAdapter or
SumoObservationAdapter being built, so both branches are written against
one frozen shape from the start instead of drifting apart independently.

Conventions this schema enforces, per the plan document's "Quy ước cần
khóa trước khi code":
  - position_unit/speed_unit are required fields, not just documentation --
    a video observation is always normalized_image/px_s, a simulation
    observation (clean or noisy) is always lane_road/m_s. analytics must
    never compare across these without an explicit, versioned transform.
  - object_id on a video observation is the tracker's own ID, not a
    physical-identity ground truth (tracks can still fragment/switch under
    occlusion -- see the `tracker:` comment in
    configs/pipeline/offline_video.yaml). object_id on sumo_clean is the
    simulator's real vehicle ID.
"""

from __future__ import annotations

import math
from typing import Any


TRAFFIC_OBSERVATION_SCHEMA_VERSION = 1

SOURCES = ("video", "sumo_clean", "sumo_noisy")
PERCEPTION_STATES = ("reliable", "degraded", "detection_silence", "unavailable")
OBSERVATION_KINDS = ("tracked", "simulator_ground_truth")
POSITION_UNITS = ("normalized_image", "lane_road")
SPEED_UNITS = ("px_s", "m_s")

_VIDEO_UNITS = ("normalized_image", "px_s")
_SUMO_UNITS = ("lane_road", "m_s")


class ObservationSchemaError(ValueError):
    """Raised when a TrafficObservation payload violates schema v1."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservationSchemaError(f"{name} must be an object")
    return value


def _exact_keys(
    payload: dict[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    missing = required - payload.keys()
    extra = payload.keys() - required
    if missing:
        raise ObservationSchemaError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise ObservationSchemaError(f"{name} has unsupported fields: {sorted(extra)}")


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationSchemaError(f"{name} must be non-empty text")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservationSchemaError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ObservationSchemaError(f"{name} must be finite")
    return number


def _optional_unit_confidence(value: Any, name: str) -> None:
    if value is None:
        return
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ObservationSchemaError(f"{name} must be in [0, 1] or null")


def validate_traffic_observation(payload: Any) -> None:
    """Validate one TrafficObservation record against schema v1."""
    observation = _mapping(payload, "traffic_observation")
    _exact_keys(
        observation,
        required={
            "schema_version",
            "source",
            "run_id",
            "timestamp_s",
            "frame_or_step",
            "position_unit",
            "speed_unit",
            "perception_status",
            "objects",
        },
        name="traffic_observation",
    )
    if observation["schema_version"] != TRAFFIC_OBSERVATION_SCHEMA_VERSION:
        raise ObservationSchemaError(
            "unsupported traffic_observation schema_version"
        )
    source = observation["source"]
    if source not in SOURCES:
        raise ObservationSchemaError(f"source must be one of {SOURCES}")
    _nonempty_text(observation["run_id"], "run_id")
    timestamp_s = _finite_number(observation["timestamp_s"], "timestamp_s")
    if timestamp_s < 0:
        raise ObservationSchemaError("timestamp_s cannot be negative")
    if (
        not isinstance(observation["frame_or_step"], int)
        or isinstance(observation["frame_or_step"], bool)
        or observation["frame_or_step"] < 0
    ):
        raise ObservationSchemaError(
            "frame_or_step must be a non-negative integer"
        )

    position_unit = observation["position_unit"]
    speed_unit = observation["speed_unit"]
    if position_unit not in POSITION_UNITS:
        raise ObservationSchemaError(f"position_unit must be one of {POSITION_UNITS}")
    if speed_unit not in SPEED_UNITS:
        raise ObservationSchemaError(f"speed_unit must be one of {SPEED_UNITS}")
    units = (position_unit, speed_unit)
    if source == "video" and units != _VIDEO_UNITS:
        raise ObservationSchemaError(
            "source=video requires position_unit=normalized_image and "
            "speed_unit=px_s -- analytics must not implicitly treat px/s "
            "as m/s"
        )
    if source in ("sumo_clean", "sumo_noisy") and units != _SUMO_UNITS:
        raise ObservationSchemaError(
            "source=sumo_clean/sumo_noisy requires position_unit=lane_road "
            "and speed_unit=m_s"
        )

    perception_status = _mapping(
        observation["perception_status"], "perception_status"
    )
    _exact_keys(
        perception_status,
        required={"state", "reason_codes", "confidence"},
        name="perception_status",
    )
    if perception_status["state"] not in PERCEPTION_STATES:
        raise ObservationSchemaError(
            f"perception_status.state must be one of {PERCEPTION_STATES}"
        )
    reason_codes = perception_status["reason_codes"]
    if not isinstance(reason_codes, list) or not all(
        isinstance(code, str) and code for code in reason_codes
    ):
        raise ObservationSchemaError(
            "perception_status.reason_codes must be a list of non-empty strings"
        )
    _optional_unit_confidence(
        perception_status["confidence"], "perception_status.confidence"
    )

    objects = observation["objects"]
    if not isinstance(objects, list):
        raise ObservationSchemaError("objects must be a list")
    seen_object_ids: set[str] = set()
    for index, item in enumerate(objects):
        record = _mapping(item, f"objects[{index}]")
        _exact_keys(
            record,
            required={
                "object_id",
                "class_name",
                "region_id",
                "position",
                "speed",
                "confidence",
                "observation_kind",
            },
            name=f"objects[{index}]",
        )
        object_id = _nonempty_text(
            record["object_id"], f"objects[{index}].object_id"
        )
        if object_id in seen_object_ids:
            raise ObservationSchemaError(f"duplicate object_id: {object_id}")
        seen_object_ids.add(object_id)
        _nonempty_text(record["class_name"], f"objects[{index}].class_name")
        if record["region_id"] is not None:
            _nonempty_text(record["region_id"], f"objects[{index}].region_id")
        position = record["position"]
        if not isinstance(position, list) or len(position) != 2:
            raise ObservationSchemaError(
                f"objects[{index}].position must be [x, y]"
            )
        for axis, value in zip("xy", position):
            _finite_number(value, f"objects[{index}].position.{axis}")
        if record["speed"] is not None:
            speed = _finite_number(record["speed"], f"objects[{index}].speed")
            if speed < 0:
                raise ObservationSchemaError(
                    f"objects[{index}].speed cannot be negative"
                )
        _optional_unit_confidence(
            record["confidence"], f"objects[{index}].confidence"
        )
        observation_kind = record["observation_kind"]
        if observation_kind not in OBSERVATION_KINDS:
            raise ObservationSchemaError(
                f"objects[{index}].observation_kind must be one of "
                f"{OBSERVATION_KINDS}"
            )
        if source == "video" and observation_kind != "tracked":
            raise ObservationSchemaError(
                f"objects[{index}].observation_kind must be 'tracked' for "
                "source=video"
            )
        if source == "sumo_clean" and observation_kind != "simulator_ground_truth":
            raise ObservationSchemaError(
                f"objects[{index}].observation_kind must be "
                "'simulator_ground_truth' for source=sumo_clean"
            )
