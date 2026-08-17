from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.contracts import (  # noqa: E402
    build_llm_request,
    build_vlm_request,
    ContractError,
    validate_llm_report,
    validate_vlm_assessment,
    validate_vlm_request,
)
from vn_traffic.reasoning.freeze import (  # noqa: E402
    build_evidence_lock,
    canonical_sha256,
    verify_evidence_lock,
)


def valid_request() -> dict:
    return {
        "schema_version": 1,
        "case_id": "evaluation-0001",
        "task": "traffic_event_visual_review",
        "locale": "vi-VN",
        "event": {
            "schema_version": 2,
            "event_id": "event-000004",
            "event_type": "congestion_transition",
            "frame_index": 51,
            "timestamp_s": 2.04,
            "previous_state": "NORMAL",
            "current_state": "CONGESTED",
            "measurements": {
                "bbox_union_occupancy": 0.61,
                "roi_track_count": 42,
            },
        },
        "evidence": {
            "keyframes": [
                {
                    "ref": "keyframe-1",
                    "path": "evidence/frames/event-000004.jpg",
                    "sha256": "a" * 64,
                }
            ],
            "clips": [],
        },
        "constraints": {
            "deterministic_fields_are_authoritative": True,
            "infer_physical_speed": False,
            "infer_event_cause": False,
        },
    }


class ReasoningContractTests(unittest.TestCase):
    def test_development_config_never_targets_evaluation_lock(self) -> None:
        config = yaml.safe_load(
            (
                PROJECT_ROOT / "configs" / "reasoning" / "development_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["split"], "development")
        self.assertIn("evidence_dev_v1", config["input_lock"])
        self.assertEqual(config["execution_policy"], "sequential_load_run_unload")

    def test_validates_cited_vlm_assessment(self) -> None:
        request = valid_request()
        assessment = {
            "schema_version": 1,
            "case_id": "evaluation-0001",
            "event_id": "event-000004",
            "observations": [
                {
                    "claim_vi": "Nhiều phương tiện xuất hiện trong hành lang đường.",
                    "confidence": 0.8,
                    "evidence_refs": ["keyframe-1"],
                }
            ],
            "incident_assessment": {
                "status": "not_observed",
                "category": "none",
                "confidence": 0.7,
            },
            "limitations": ["Một khung hình không chứng minh nguyên nhân ùn tắc."],
        }

        validate_vlm_assessment(assessment, request)

        invalid = copy.deepcopy(assessment)
        invalid["observations"][0]["evidence_refs"] = ["unknown-frame"]
        with self.assertRaisesRegex(ContractError, "unknown evidence"):
            validate_vlm_assessment(invalid, request)

    def test_rejects_relaxed_safety_constraint(self) -> None:
        request = valid_request()
        request["constraints"]["infer_event_cause"] = True
        with self.assertRaisesRegex(ContractError, "must not be relaxed"):
            validate_vlm_request(request)

    def test_llm_numeric_facts_must_equal_deterministic_event(self) -> None:
        request = valid_request()
        assessment = {
            "schema_version": 1,
            "case_id": "evaluation-0001",
            "event_id": "event-000004",
            "observations": [],
            "incident_assessment": {
                "status": "uncertain",
                "category": "none",
                "confidence": 0.4,
            },
            "limitations": [],
        }
        llm_request = build_llm_request(request, assessment)
        report = {
            "schema_version": 1,
            "case_id": "evaluation-0001",
            "event_id": "event-000004",
            "summary_vi": "Hệ thống ghi nhận chuyển trạng thái sang ùn tắc.",
            "traffic_state": "CONGESTED",
            "numeric_facts": [
                {
                    "source_path": "event.measurements.roi_track_count",
                    "value": 42,
                }
            ],
            "visual_findings": ["Mật độ phương tiện quan sát được ở mức cao."],
            "action": {"level": "monitor", "message_vi": "Tiếp tục theo dõi."},
            "limitations": ["Chưa có hiệu chuẩn tốc độ vật lý."],
        }

        validate_llm_report(report, llm_request)

        report["numeric_facts"][0]["value"] = 43
        with self.assertRaisesRegex(ContractError, "differs from deterministic"):
            validate_llm_report(report, llm_request)

    def test_builds_request_from_frozen_case(self) -> None:
        case = {
            "case_id": "evaluation-0001",
            "event": valid_request()["event"],
            "evidence": {
                "keyframe": {"path": "frame.jpg", "sha256": "b" * 64}
            },
        }
        request = build_vlm_request(case)
        self.assertEqual(request["evidence"]["keyframes"][0]["ref"], "keyframe-1")
        validate_vlm_request(request)


class EvidenceLockTests(unittest.TestCase):
    def test_checked_in_evaluation_lock_is_valid(self) -> None:
        path = (
            PROJECT_ROOT
            / "manifests"
            / "reasoning"
            / "evidence_eval_v1"
            / "input_lock.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        verify_evidence_lock(payload)
        self.assertEqual(payload["case_count"], 14)
        self.assertEqual(payload["split"], "evaluation")
        self.assertEqual(
            {case["event"]["event_type"] for case in payload["cases"]},
            {"line_crossing", "congestion_transition"},
        )

    def test_checked_in_development_lock_is_source_disjoint(self) -> None:
        evaluation = json.loads(
            (
                PROJECT_ROOT
                / "manifests"
                / "reasoning"
                / "evidence_eval_v1"
                / "input_lock.json"
            ).read_text(encoding="utf-8")
        )
        development = json.loads(
            (
                PROJECT_ROOT
                / "manifests"
                / "reasoning"
                / "evidence_dev_v1"
                / "input_lock.json"
            ).read_text(encoding="utf-8")
        )
        verify_evidence_lock(development)
        self.assertEqual(development["split"], "development")
        self.assertEqual(development["case_count"], 146)
        self.assertNotEqual(
            development["source"]["source_video_sha256"],
            evaluation["source"]["source_video_sha256"],
        )

    def test_builds_and_detects_tampered_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            source = run_dir / "source.mp4"
            source.write_bytes(b"source-video")
            artifact = run_dir / "evidence" / "frames" / "event-1.jpg"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"keyframe")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            event = {
                "schema_version": 2,
                "event_id": "event-1",
                "event_type": "line_crossing",
                "frame_index": 2,
                "timestamp_s": 0.2,
            }
            evidence = {
                "schema_version": 2,
                "evidence_id": "evidence-event-1",
                "event_id": "event-1",
                "event_type": "line_crossing",
                "source_video_sha256": source_sha,
                "source_frame_index": 2,
                "source_timestamp_s": 0.2,
                "keyframe": {
                    "path": "evidence/frames/event-1.jpg",
                    "sha256": artifact_sha,
                },
            }
            (run_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            (run_dir / "evidence.jsonl").write_text(
                json.dumps(evidence) + "\n", encoding="utf-8"
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "run-test",
                        "source": str(source),
                        "frames_processed": 3,
                        "evidence": {
                            "schema_version": 2,
                            "source_video_sha256": source_sha,
                        },
                    }
                ),
                encoding="utf-8",
            )

            lock = build_evidence_lock(
                run_dir=run_dir,
                set_id="test-v1",
                split="evaluation",
            )
            verify_evidence_lock(lock)
            self.assertEqual(lock["case_count"], 1)
            self.assertEqual(lock["cases"][0]["event_sha256"], canonical_sha256(event))

            lock["cases"][0]["event"]["frame_index"] = 99
            with self.assertRaisesRegex(ValueError, "lock SHA-256 mismatch"):
                verify_evidence_lock(lock)


if __name__ == "__main__":
    unittest.main()
