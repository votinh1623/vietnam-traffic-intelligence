from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from audit_visual_overlap import difference_hash, hamming  # noqa: E402


class VisualOverlapTests(unittest.TestCase):
    def test_identical_patterns_have_zero_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
            first = root / "first.png"
            second = root / "second.png"
            cv2.imwrite(str(first), image)
            cv2.imwrite(str(second), image)
            self.assertEqual(hamming(difference_hash(first), difference_hash(second)), 0)


if __name__ == "__main__":
    unittest.main()

