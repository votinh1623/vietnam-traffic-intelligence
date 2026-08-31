from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.measurement_manifest import (  # noqa: E402
    MeasurementManifestError,
    load_measurement_manifest,
    validate_measurement_manifest,
)


def manifest(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "scene_id": "video_7938",
        "camera": {"motion": "moving"},
        "measurement": {
            "roi_polygon": [[0.48, 0.05], [0.62, 0.05], [0.84, 0.95], [0.30, 0.95]],
            "ignore_regions": [],
            "regions": [
                {
                    "id": "region_1",
                    "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                }
            ],
            "counting_lines": [
                {
                    "id": "line_1",
                    "points": [[0.32, 0.65], [0.78, 0.65]],
                    "direction_labels": {
                        "side_a_to_b": "direction_a",
                        "side_b_to_a": "direction_b",
                    },
                }
            ],
        },
        "provenance": {
            "author": "manual",
            "created_at": "2026-08-27T00:00:00+00:00",
            "source_sha256": "a" * 64,
        },
    }
    base.update(overrides)
    return base


class MeasurementManifestSchemaTests(unittest.TestCase):
    def test_accepts_a_valid_manifest(self) -> None:
        validate_measurement_manifest(manifest())

    def test_accepts_null_roi_polygon_as_full_frame(self) -> None:
        payload = manifest()
        payload["measurement"] = {**payload["measurement"], "roi_polygon": None}
        validate_measurement_manifest(payload)

    def test_rejects_unnormalized_coordinates(self) -> None:
        payload = manifest()
        payload["measurement"]["roi_polygon"][0] = [1.5, 0.05]
        with self.assertRaisesRegex(MeasurementManifestError, "normalized"):
            validate_measurement_manifest(payload)

    def test_rejects_duplicate_region_ids(self) -> None:
        payload = manifest()
        payload["measurement"]["regions"].append(payload["measurement"]["regions"][0])
        with self.assertRaisesRegex(MeasurementManifestError, "duplicate region id"):
            validate_measurement_manifest(payload)

    def test_rejects_duplicate_counting_line_ids(self) -> None:
        payload = manifest()
        payload["measurement"]["counting_lines"].append(
            payload["measurement"]["counting_lines"][0]
        )
        with self.assertRaisesRegex(
            MeasurementManifestError, "duplicate counting_line id"
        ):
            validate_measurement_manifest(payload)

    def test_rejects_counting_line_with_identical_endpoints(self) -> None:
        payload = manifest()
        payload["measurement"]["counting_lines"][0]["points"] = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        with self.assertRaisesRegex(MeasurementManifestError, "must differ"):
            validate_measurement_manifest(payload)

    def test_rejects_unknown_camera_motion(self) -> None:
        payload = manifest()
        payload["camera"] = {"motion": "orbiting"}
        with self.assertRaisesRegex(MeasurementManifestError, "camera.motion"):
            validate_measurement_manifest(payload)

    def test_rejects_invalid_source_sha256(self) -> None:
        payload = manifest()
        payload["provenance"]["source_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(MeasurementManifestError, "source_sha256"):
            validate_measurement_manifest(payload)

    def test_rejects_extra_top_level_field(self) -> None:
        payload = manifest()
        payload["extra"] = True
        with self.assertRaisesRegex(MeasurementManifestError, "unsupported fields"):
            validate_measurement_manifest(payload)

    def test_load_measurement_manifest_reads_and_validates_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.yaml"
            import yaml

            path.write_text(yaml.safe_dump(manifest()), encoding="utf-8")
            loaded = load_measurement_manifest(path)
            self.assertEqual(loaded["scene_id"], "video_7938")

    def test_load_measurement_manifest_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_measurement_manifest(Path("does/not/exist.yaml"))


if __name__ == "__main__":
    unittest.main()
