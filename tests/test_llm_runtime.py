from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from vn_traffic.reasoning.contracts import ContractError, build_vlm_request
from vn_traffic.reasoning.llm_runtime import (
    assemble_llm_report,
    build_report_prompt,
    prepare_llm_request,
)


def _request() -> dict:
    return build_vlm_request(
        {
            "case_id": "development-0001",
            "event": {
                "schema_version": 2,
                "event_id": "event-000001",
                "event_type": "line_crossing",
                "frame_index": 10,
                "timestamp_s": 1.0,
                "measurements": {"speed_px_s": 12.5},
            },
            "evidence": {
                "schema_version": 2,
                "evidence_id": "evidence-event-000001",
                "event_id": "event-000001",
                "event_type": "line_crossing",
                "source_video_sha256": "a" * 64,
                "source_frame_index": 10,
                "source_timestamp_s": 1.0,
                "keyframe": {"path": "frame.jpg", "sha256": "b" * 64},
            },
        }
    )


def _result() -> dict:
    return {
        "case_id": "development-0001",
        "contract_status": "valid",
        "assessment": {
            "schema_version": 1,
            "case_id": "development-0001",
            "event_id": "event-000001",
            "observations": [],
            "incident_assessment": {
                "status": "uncertain",
                "category": "none",
                "confidence": 0.5,
            },
            "limitations": [],
        },
    }


class LlmRuntimeTests(unittest.TestCase):
    def _write(self, payload: dict, directory: str) -> Path:
        path = Path(directory) / "vlm.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_prepares_request_only_from_validated_matching_vlm_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, request = prepare_llm_request(
                self._write(_result(), directory), _request()
            )
        prompt = build_report_prompt(request)
        self.assertIn("containing only summary_vi and action", prompt)
        self.assertNotIn("Tóm tắt thận trọng", prompt)
        self.assertNotIn("Nêu giới hạn của evidence", prompt)

        report = assemble_llm_report(
            {
                "summary_vi": "Ghi nhận một sự kiện giao thông cần theo dõi.",
                "action": {"level": "monitor", "message_vi": "Tiếp tục theo dõi."},
            },
            request,
        )
        self.assertEqual(report["traffic_state"], "UNSPECIFIED")
        self.assertEqual(
            report["numeric_facts"],
            [{"source_path": "event.measurements.speed_px_s", "value": 12.5}],
        )

    def test_assembler_rejects_model_owned_authoritative_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, request = prepare_llm_request(
                self._write(_result(), directory), _request()
            )
        with self.assertRaisesRegex(ContractError, "only summary_vi and action"):
            assemble_llm_report(
                {
                    "summary_vi": "Báo cáo.",
                    "action": {"level": "none", "message_vi": "Không hành động."},
                    "numeric_facts": [],
                },
                request,
            )

    def test_rejects_vlm_result_marked_invalid(self) -> None:
        result = _result()
        result["contract_status"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContractError, "contract_status=valid"):
                prepare_llm_request(self._write(result, directory), _request())

    def test_rejects_mismatched_or_mutated_assessment(self) -> None:
        result = _result()
        result["case_id"] = "development-0002"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContractError, "case_id"):
                prepare_llm_request(self._write(result, directory), _request())

        result = copy.deepcopy(_result())
        result["assessment"]["event_id"] = "event-000002"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContractError, "event_id"):
                prepare_llm_request(self._write(result, directory), _request())


if __name__ == "__main__":
    unittest.main()
