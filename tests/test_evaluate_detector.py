from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_detector import validate_config  # noqa: E402


class EvaluateDetectorTests(unittest.TestCase):
    def test_locked_evaluation_requires_test_split(self) -> None:
        config = {
            "status": "frozen_final_test",
            "dataset": {"split": "validation"},
            "provenance": {"require_clean_worktree": True},
        }
        with self.assertRaisesRegex(ValueError, "split: test"):
            validate_config(config)

    def test_locked_evaluation_requires_clean_worktree_policy(self) -> None:
        config = {
            "status": "frozen_final_test",
            "dataset": {"split": "test"},
            "provenance": {"require_clean_worktree": False},
        }
        with self.assertRaisesRegex(ValueError, "clean worktree"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
