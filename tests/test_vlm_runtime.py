from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


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

    def test_prompt_example_is_not_copyable_vietnamese_prose(self) -> None:
        # Regression guard: every run before this fix reproduced the old
        # literal claim_vi example verbatim (see output/reasoning/adhoc/*.json
        # and output/reasoning/dev_v1/*.json) because it was valid-looking
        # Vietnamese prose that a small model could just copy as its answer.
        _, request, _ = load_development_case(
            PROJECT_ROOT / "configs" / "reasoning" / "development_v1.yaml",
            "development-0001",
        )
        prompt = _prompt_text(request)
        self.assertNotIn("Quan sát thấy các phương tiện trong khung hình", prompt)
        self.assertIn("KHONG duoc chep nguyen van", prompt)

    def test_system_prompt_v3_has_no_copyable_example_sentence(self) -> None:
        # Regression test: prompts_v2.yaml's system prompt illustrated the
        # required structure with a complete, valid Vietnamese sentence
        # ("chủ yếu là xe máy, có nhiều ô tô con và vài xe buýt hoặc xe
        # tải"), and the VLM copied it verbatim onto an unrelated,
        # truck-dominated keyframe -- see
        # output/reasoning/adhoc/run34-vlm-v2prompt.json, where the analytics
        # for that same frame recorded car:17, motorcycle:1, truck:46. v3
        # replaces it with a fill-in-the-brackets template.
        prompts = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "reasoning" / "prompts_v3.yaml").read_text(
                encoding="utf-8"
            )
        )
        system_prompt = prompts["vlm"]["system"]
        self.assertNotIn(
            "chủ yếu là xe máy, có nhiều ô tô con và vài xe buýt hoặc xe tải",
            system_prompt,
        )
        self.assertIn("[LOẠI XE CHIẾM ĐA SỐ", system_prompt)
        self.assertIn("[THƯA HOẶC VỪA", system_prompt)

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
