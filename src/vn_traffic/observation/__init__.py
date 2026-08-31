"""Common observation contract shared by the real-video and SUMO branches."""

from .schema import (
    ObservationSchemaError,
    OBSERVATION_KINDS,
    PERCEPTION_STATES,
    POSITION_UNITS,
    SOURCES,
    SPEED_UNITS,
    TRAFFIC_OBSERVATION_SCHEMA_VERSION,
    validate_traffic_observation,
)

__all__ = [
    "ObservationSchemaError",
    "OBSERVATION_KINDS",
    "PERCEPTION_STATES",
    "POSITION_UNITS",
    "SOURCES",
    "SPEED_UNITS",
    "TRAFFIC_OBSERVATION_SCHEMA_VERSION",
    "validate_traffic_observation",
]
