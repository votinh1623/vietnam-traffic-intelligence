"""Measurement manifest schema v1.

Per-video measurement geometry only -- never traffic-law inference. See
reports/ke-hoach-pipeline-va-mo-phong.md section 2.3: video sources can come
from unknown locations/angles/times, so the system must not guess lane
rules, right-turn-on-red, or signal phase from image content. A manifest
describes only where to measure (ROI, named regions, counting lines with
neutral direction labels), never what the traffic rules there are.

All polygon/line points are normalized image coordinates in [0, 1], the
same convention `AnalyticsConfig.roi_polygon`/`counting_line` already use
(src/vn_traffic/config.py) and the same convention
`observation.schema.POSITION_UNITS`'s `"normalized_image"` names for
source="video" (src/vn_traffic/observation/schema.py).

A manifest is optional: per Gate G1, its absence must not fail the
pipeline, must not synthesize line counts, must not compare regions, and
must not produce any per-region claim -- see PipelineConfig.measurement_manifest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


MEASUREMENT_MANIFEST_SCHEMA_VERSION = 1
CAMERA_MOTIONS = ("static", "moving", "unknown")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MeasurementManifestError(ValueError):
    """Raised when a measurement manifest payload violates schema v1."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MeasurementManifestError(f"{name} must be an object")
    return value


def _exact_keys(payload: dict[str, Any], *, required: set[str], name: str) -> None:
    missing = required - payload.keys()
    extra = payload.keys() - required
    if missing:
        raise MeasurementManifestError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise MeasurementManifestError(f"{name} has unsupported fields: {sorted(extra)}")


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeasurementManifestError(f"{name} must be non-empty text")
    return value


def _normalized_point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise MeasurementManifestError(f"{name} must be [x, y]")
    coordinates = []
    for axis, component in zip("xy", value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise MeasurementManifestError(f"{name}.{axis} must be numeric")
        number = float(component)
        if not 0.0 <= number <= 1.0:
            raise MeasurementManifestError(
                f"{name}.{axis} must be normalized to [0, 1]"
            )
        coordinates.append(number)
    return (coordinates[0], coordinates[1])


def _polygon(value: Any, name: str, *, min_points: int = 3) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < min_points:
        raise MeasurementManifestError(
            f"{name} must be a list of at least {min_points} [x, y] points"
        )
    return [_normalized_point(point, f"{name}[{index}]") for index, point in enumerate(value)]


def validate_measurement_manifest(payload: Any) -> None:
    """Validate one measurement manifest against schema v1."""
    manifest = _mapping(payload, "measurement_manifest")
    _exact_keys(
        manifest,
        required={"schema_version", "scene_id", "camera", "measurement", "provenance"},
        name="measurement_manifest",
    )
    if manifest["schema_version"] != MEASUREMENT_MANIFEST_SCHEMA_VERSION:
        raise MeasurementManifestError(
            "unsupported measurement_manifest schema_version"
        )
    _nonempty_text(manifest["scene_id"], "scene_id")

    camera = _mapping(manifest["camera"], "camera")
    _exact_keys(camera, required={"motion"}, name="camera")
    if camera["motion"] not in CAMERA_MOTIONS:
        raise MeasurementManifestError(f"camera.motion must be one of {CAMERA_MOTIONS}")

    measurement = _mapping(manifest["measurement"], "measurement")
    _exact_keys(
        measurement,
        required={"roi_polygon", "ignore_regions", "regions", "counting_lines"},
        name="measurement",
    )
    if measurement["roi_polygon"] is not None:
        _polygon(measurement["roi_polygon"], "measurement.roi_polygon")

    ignore_regions = measurement["ignore_regions"]
    if not isinstance(ignore_regions, list):
        raise MeasurementManifestError("measurement.ignore_regions must be a list")
    for index, region in enumerate(ignore_regions):
        _polygon(region, f"measurement.ignore_regions[{index}]")

    regions = measurement["regions"]
    if not isinstance(regions, list):
        raise MeasurementManifestError("measurement.regions must be a list")
    seen_region_ids: set[str] = set()
    for index, region in enumerate(regions):
        record = _mapping(region, f"measurement.regions[{index}]")
        _exact_keys(record, required={"id", "polygon"}, name=f"measurement.regions[{index}]")
        region_id = _nonempty_text(record["id"], f"measurement.regions[{index}].id")
        if region_id in seen_region_ids:
            raise MeasurementManifestError(f"duplicate region id: {region_id}")
        seen_region_ids.add(region_id)
        _polygon(record["polygon"], f"measurement.regions[{index}].polygon")

    counting_lines = measurement["counting_lines"]
    if not isinstance(counting_lines, list):
        raise MeasurementManifestError("measurement.counting_lines must be a list")
    seen_line_ids: set[str] = set()
    for index, line in enumerate(counting_lines):
        record = _mapping(line, f"measurement.counting_lines[{index}]")
        _exact_keys(
            record,
            required={"id", "points", "direction_labels"},
            name=f"measurement.counting_lines[{index}]",
        )
        line_id = _nonempty_text(record["id"], f"measurement.counting_lines[{index}].id")
        if line_id in seen_line_ids:
            raise MeasurementManifestError(f"duplicate counting_line id: {line_id}")
        seen_line_ids.add(line_id)
        points = record["points"]
        if not isinstance(points, list) or len(points) != 2:
            raise MeasurementManifestError(
                f"measurement.counting_lines[{index}].points must be exactly two "
                "[x, y] points"
            )
        for point_index, point in enumerate(points):
            _normalized_point(
                point, f"measurement.counting_lines[{index}].points[{point_index}]"
            )
        if points[0] == points[1]:
            raise MeasurementManifestError(
                f"measurement.counting_lines[{index}].points must differ"
            )
        labels = _mapping(
            record["direction_labels"],
            f"measurement.counting_lines[{index}].direction_labels",
        )
        _exact_keys(
            labels,
            required={"side_a_to_b", "side_b_to_a"},
            name=f"measurement.counting_lines[{index}].direction_labels",
        )
        _nonempty_text(
            labels["side_a_to_b"],
            f"measurement.counting_lines[{index}].direction_labels.side_a_to_b",
        )
        _nonempty_text(
            labels["side_b_to_a"],
            f"measurement.counting_lines[{index}].direction_labels.side_b_to_a",
        )

    provenance = _mapping(manifest["provenance"], "provenance")
    _exact_keys(
        provenance,
        required={"author", "created_at", "source_sha256"},
        name="provenance",
    )
    _nonempty_text(provenance["author"], "provenance.author")
    _nonempty_text(provenance["created_at"], "provenance.created_at")
    source_sha256 = provenance["source_sha256"]
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise MeasurementManifestError(
            "provenance.source_sha256 must be lowercase SHA-256"
        )


def load_measurement_manifest(path: Path) -> dict[str, Any]:
    """Load and validate one measurement manifest YAML file."""
    import yaml

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"measurement manifest not found: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    validate_measurement_manifest(payload)
    return payload
