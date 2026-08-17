from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.annotations import (  # noqa: E402
    build_adjudication_queue,
    build_annotation_template,
    build_review_index,
    validate_annotation_set,
)


LOCK_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "reasoning"
    / "evidence_eval_v1"
    / "input_lock.json"
)


def completed(records: list[dict]) -> list[dict]:
    result = copy.deepcopy(records)
    for record in result:
        record.update(
            {
                "annotation_status": "complete",
                "evidence_quality": "clear",
                "visible_density": "high",
                "visible_classes": ["motorcycle", "car"],
                "event_visual_support": "insufficient",
                "incident_status": "not_observed",
                "incident_category": "none",
                "observations_vi": ["Có nhiều phương tiện trong vùng quan sát."],
                "reference_summary_vi": "Khung hình cho thấy giao thông đông.",
                "required_limitations_vi": [
                    "Một keyframe không chứng minh hướng di chuyển."
                ],
                "notes_vi": "",
            }
        )
    return result


class ReasoningAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_template_has_exact_frozen_case_coverage(self) -> None:
        records = build_annotation_template(self.lock, "reviewer_a")
        reviewer = validate_annotation_set(
            records, self.lock, require_complete=False
        )
        self.assertEqual(reviewer, "reviewer_a")
        self.assertEqual(len(records), 14)
        with self.assertRaisesRegex(ValueError, "still pending"):
            validate_annotation_set(records, self.lock, require_complete=True)

    def test_review_index_resolves_local_evidence_paths(self) -> None:
        rows = build_review_index(self.lock)
        self.assertEqual(len(rows), 14)
        self.assertTrue(rows[0]["keyframe_path"].startswith("output/pipeline/run15/"))
        congestion = [row for row in rows if row["event_type"] == "congestion_transition"]
        self.assertEqual(len(congestion), 1)
        self.assertTrue(congestion[0]["clip_path"].endswith(".mp4"))

    def test_complete_annotation_enforces_incident_consistency(self) -> None:
        records = completed(build_annotation_template(self.lock, "reviewer_a"))
        validate_annotation_set(records, self.lock, require_complete=True)
        records[0]["incident_category"] = "collision"
        with self.assertRaisesRegex(ValueError, "must use category none"):
            validate_annotation_set(records, self.lock, require_complete=True)

    def test_rejects_missing_or_reordered_case(self) -> None:
        records = build_annotation_template(self.lock, "reviewer_a")
        records.reverse()
        with self.assertRaisesRegex(ValueError, "exactly match lock"):
            validate_annotation_set(records, self.lock, require_complete=False)

    def test_adjudication_queue_reports_categorical_disagreement(self) -> None:
        first = completed(build_annotation_template(self.lock, "reviewer_a"))
        second = completed(build_annotation_template(self.lock, "reviewer_b"))
        second[0]["visible_density"] = "medium"
        second[0]["reference_summary_vi"] = "Mật độ quan sát ở mức vừa."

        queue = build_adjudication_queue(first, second, self.lock)

        self.assertEqual(queue["case_count"], 14)
        self.assertEqual(queue["categorical_disagreement_cases"], 1)
        self.assertIn(
            "visible_density", queue["cases"][0]["categorical_disagreements"]
        )
        self.assertEqual(queue["cases"][0]["adjudication_status"], "pending")

    def test_adjudication_requires_distinct_reviewers(self) -> None:
        first = completed(build_annotation_template(self.lock, "reviewer_a"))
        second = copy.deepcopy(first)
        with self.assertRaisesRegex(ValueError, "distinct reviewers"):
            build_adjudication_queue(first, second, self.lock)


if __name__ == "__main__":
    unittest.main()
