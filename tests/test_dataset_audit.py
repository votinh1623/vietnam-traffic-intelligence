from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from audit_dataset import (  # noqa: E402
    audit_manifest,
    build_manifest,
    discover_split_dirs,
    parse_source_frame,
)


class DatasetAuditTests(unittest.TestCase):
    def test_parse_roboflow_frame_name(self) -> None:
        source, frame = parse_source_frame(
            "traffic_jam_frame_00006_jpg.rf.0ca9d8c442207acb033bea0125317f58.jpg"
        )
        self.assertEqual(source, "traffic_jam")
        self.assertEqual(frame, 6)

    def test_manifest_detects_group_and_same_frame_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)

            for split, frame in (("train", 1), ("valid", 1), ("test", 2)):
                name = f"video_a_frame_{frame:05d}.jpg"
                (root / split / "images" / name).write_bytes(f"{split}-{frame}".encode())
                (root / split / "labels" / f"video_a_frame_{frame:05d}.txt").write_text(
                    "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )

            rows = build_manifest(root)
            audit = audit_manifest(rows)
            self.assertEqual(audit["status"], "invalid_leakage")
            self.assertEqual(audit["overlapping_source_count"], 1)
            self.assertEqual(audit["same_source_frame_cross_split_count"], 1)
            self.assertEqual(
                audit["validation_vs_train_temporal_gap_frames"]["minimum"], 0
            )
            self.assertEqual(audit["test_vs_train_temporal_gap_frames"]["minimum"], 1)

    def test_polygon_labels_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "labels").mkdir(parents=True)
            name = "video_b_frame_00001.jpg"
            (root / "train" / "images" / name).write_bytes(b"image")
            (root / "train" / "labels" / "video_b_frame_00001.txt").write_text(
                "2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
            )
            rows = build_manifest(root)
            self.assertEqual(rows[0]["label_status"], "labeled")
            self.assertEqual(rows[0]["annotation_formats"], "polygon:1")

    def test_normalized_split_names_include_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "calibration", "validation", "test"):
                (root / split).mkdir()
            self.assertEqual(
                discover_split_dirs(root),
                {
                    "train": "train",
                    "calibration": "calibration",
                    "validation": "validation",
                    "test": "test",
                },
            )


if __name__ == "__main__":
    unittest.main()
