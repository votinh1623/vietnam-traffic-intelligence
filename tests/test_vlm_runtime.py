from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.vlm_runtime import (  # noqa: E402
    _prompt_text,
    extract_json_object,
    load_development_case,
    validate_grounding_policy,
)


class VLMRuntimeTests(unittest.TestCase):
    def test_extracts_single_fenced_json_object(self) -> None:
        payload = extract_json_object('```json\n{"schema_version": 1}\n```')
        self.assertEqual(payload, {"schema_version": 1})

    def test_rejects_text_after_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON object"):
            extract_json_object('{"schema_version": 1} unsupported explanation')

    def test_prompt_example_does_not_prime_zero_confidence(self) -> None:
        _, request, _ = load_development_case(
            PROJECT_ROOT / "configs" / "reasoning" / "development_v1.yaml",
            "development-0001",
        )
        prompt = _prompt_text(request)
        self.assertNotIn('"confidence": 0.0', prompt)
        self.assertNotIn('"direction"', prompt)
        self.assertNotIn('"speed_px_s"', prompt)
        self.assertNotIn('"class_name"', prompt)

    def test_loads_hashed_development_case_only(self) -> None:
        config, request, artifact_root = load_development_case(
            PROJECT_ROOT / "configs" / "reasoning" / "development_v1.yaml",
            "development-0001",
        )
        self.assertEqual(config["split"], "development")
        self.assertEqual(request["case_id"], "development-0001")
        self.assertTrue(
            (artifact_root / request["evidence"]["keyframes"][0]["path"]).is_file()
        )

    def test_rejects_motion_claim_from_keyframe_only_evidence(self) -> None:
        _, request, _ = load_development_case(
            PROJECT_ROOT / "configs" / "reasoning" / "development_v1.yaml",
            "development-0001",
        )
        assessment = {
            "observations": [
                {"claim_vi": "Xe máy đang di chuyển qua giao lộ."}
            ]
        }
        with self.assertRaisesRegex(ValueError, "motion from keyframe"):
            validate_grounding_policy(assessment, request)


if __name__ == "__main__":
    unittest.main()
