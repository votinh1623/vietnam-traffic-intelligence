from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from lock_test_split import build_lock, verify_lock  # noqa: E402


class LockTestSplitTests(unittest.TestCase):
    def test_lock_only_contains_test_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for split in ("train", "test"):
                image = root / f"{split}.jpg"
                label = root / f"{split}.txt"
                image.write_bytes(split.encode())
                label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                rows.append(
                    {
                        "source_id": split,
                        "frame_index": "1",
                        "split": split,
                        "image_path": image.name,
                        "label_path": label.name,
                        "bbox_count": "1",
                    }
                )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            first = build_lock(root, manifest)
            second = build_lock(root, manifest)
            self.assertEqual(first["image_count"], 1)
            self.assertEqual(first["items"][0]["source_id"], "test")
            self.assertEqual(first["lock_sha256"], second["lock_sha256"])

            lock_path = root / "test_lock.json"
            import json

            lock_path.write_text(json.dumps(first), encoding="utf-8")
            self.assertTrue(verify_lock(root, lock_path)["valid"])
            (root / "test.txt").write_text("changed\n", encoding="utf-8")
            verification = verify_lock(root, lock_path)
            self.assertFalse(verification["valid"])
            self.assertTrue(any("label hash mismatch" in e for e in verification["errors"]))


if __name__ == "__main__":
    unittest.main()
