from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from materialize_v4 import (  # noqa: E402
    convert_annotation_line,
    materialize,
    select_records,
)


class MaterializeV4Tests(unittest.TestCase):
    def test_polygon_is_converted_to_bbox(self) -> None:
        converted, source_format = convert_annotation_line(
            "2 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6"
        )
        self.assertEqual(source_format, "polygon")
        self.assertEqual(converted, "2 0.2 0.4 0.2 0.4")

    def test_conflicting_duplicate_frame_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, label in enumerate(
                ("1 0.5 0.5 0.2 0.2\n", "1 0.5 0.5 0.4 0.4\n")
            ):
                (root / f"label{index}.txt").write_text(label, encoding="utf-8")
            rows = [
                {
                    "image_id": str(index),
                    "image_path": f"image{index}.jpg",
                    "label_path": f"label{index}.txt",
                    "source_id": "video",
                    "frame_index": "1",
                    "content_sha256": str(index),
                }
                for index in range(2)
            ]
            selected, exclusions = select_records(rows, root, {"video": "train"})
            self.assertEqual(selected, [])
            self.assertEqual(exclusions[0]["reason"], "duplicate_frame_label_conflict")

    def test_materialized_splits_are_source_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "v4"
            source_root.mkdir()
            selected = []
            for source, split, frame in (("a", "train", "1"), ("b", "test", "2")):
                image = source_root / f"{source}.jpg"
                image.write_bytes(source.encode())
                selected.append(
                    {
                        "source_id": source,
                        "frame_index": frame,
                        "target_split": split,
                        "source_image_path": image.name,
                        "source_label_path": "unused.txt",
                        "source_content_sha256": source,
                        "converted_label": "1 0.5 0.5 0.2 0.2\n",
                        "bbox_count": 1,
                        "source_bbox_count": 1,
                        "source_polygon_count": 0,
                        "duplicate_candidates": 1,
                    }
                )
            report = materialize(selected, [], source_root, output_root)
            self.assertEqual(report["split_counts"], {"test": 1, "train": 1})
            with (output_root / "manifest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            source_splits = {(row["source_id"], row["split"]) for row in rows}
            self.assertEqual(source_splits, {("a", "train"), ("b", "test")})


if __name__ == "__main__":
    unittest.main()

