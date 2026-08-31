from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.vlm_runtime import (  # noqa: E402
    _multi_view_note,
    _prompt_text,
    extract_json_object,
    load_development_case,
    load_prompts,
    validate_grounding_policy,
)

# Self-contained fixture (tests/fixtures/reasoning/), not the real
# configs/reasoning/development_v1.yaml -- that one's artifact_root points
# at a gitignored output/pipeline/run16, which does not exist on a fresh
# checkout.
DEVELOPMENT_CONFIG = (
    PROJECT_ROOT / "tests" / "fixtures" / "reasoning" / "development_v1.yaml"
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
            DEVELOPMENT_CONFIG,
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
            DEVELOPMENT_CONFIG,
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

    def test_cli_config_loads_declared_prompt_version_not_v1(self) -> None:
        # Regression guard: run_vlm.py/run_llm.py used to hardcode
        # prompts_v1.yaml regardless of what a config declared, silently
        # bypassing the prompt-copying fix in prompts_v3.yaml (see
        # docs/reasoning_protocol.md). development_v1.yaml now declares
        # `prompts: prompts_v3.yaml`; assert the loader actually resolves
        # to v3's content, not v1's.
        config, _, _ = load_development_case(
            DEVELOPMENT_CONFIG,
            "development-0001",
        )
        self.assertEqual(config["prompts"], "prompts_v3.yaml")
        prompts = load_prompts(config, PROJECT_ROOT)
        self.assertIn("[LOẠI XE CHIẾM ĐA SỐ", prompts["vlm"]["system"])

    def test_load_prompts_requires_declared_prompts_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompts"):
            load_prompts({}, PROJECT_ROOT)

    def test_loads_hashed_development_case_only(self) -> None:
        config, request, artifact_root = load_development_case(
            DEVELOPMENT_CONFIG,
            "development-0001",
        )
        self.assertEqual(config["split"], "development")
        self.assertEqual(request["case_id"], "development-0001")
        self.assertTrue(
            (artifact_root / request["evidence"]["keyframes"][0]["path"]).is_file()
        )

    def test_rejects_motion_claim_from_keyframe_only_evidence(self) -> None:
        _, request, _ = load_development_case(
            DEVELOPMENT_CONFIG,
            "development-0001",
        )
        assessment = {
            "observations": [
                {"claim_vi": "Xe máy đang di chuyển qua giao lộ."}
            ]
        }
        with self.assertRaisesRegex(ValueError, "motion from keyframe"):
            validate_grounding_policy(assessment, request, clip_frames_shown=False)

    def test_rejects_motion_claim_even_when_request_references_clip_evidence(
        self,
    ) -> None:
        # Regression guard: build_vlm_request adds a "clip-1" evidence ref
        # whenever the frozen case has clip evidence on disk, but
        # run_vlm_case never actually loads clip frames -- it is
        # keyframe-only regardless. A request that merely *references*
        # clip evidence must not be enough to waive the motion-claim check;
        # only clip_frames_shown=True (set by the caller once it really
        # feeds clip frames to the model) may waive it.
        _, request, _ = load_development_case(
            DEVELOPMENT_CONFIG,
            "development-0001",
        )
        request = dict(request)
        request["evidence"] = dict(request["evidence"])
        request["evidence"]["clips"] = [
            {"ref": "clip-1", "path": "evidence/clips/fake.mp4", "sha256": "0" * 64}
        ]
        assessment = {
            "observations": [
                {"claim_vi": "Xe máy đang di chuyển qua giao lộ."}
            ]
        }
        with self.assertRaisesRegex(ValueError, "motion from keyframe"):
            validate_grounding_policy(assessment, request, clip_frames_shown=False)

    def test_multi_view_note_empty_for_single_keyframe_request(self) -> None:
        _, request, _ = load_development_case(DEVELOPMENT_CONFIG, "development-0001")
        self.assertEqual(len(request["evidence"]["keyframes"]), 1)
        self.assertEqual(_multi_view_note(request), "")

    def test_multi_view_note_explains_same_instant_multiple_crops(self) -> None:
        _, request, _ = load_development_case(DEVELOPMENT_CONFIG, "development-0001")
        request = dict(request)
        request["evidence"] = dict(request["evidence"])
        request["evidence"]["keyframes"] = list(request["evidence"]["keyframes"]) + [
            {"ref": "keyframe-2", "path": "roi.jpg", "sha256": "0" * 64},
            {"ref": "keyframe-3", "path": "event.jpg", "sha256": "0" * 64},
        ]
        note = _multi_view_note(request)
        self.assertIn("3 images", note)
        self.assertIn("keyframe-1, keyframe-2, keyframe-3", note)
        self.assertIn("SAME instant", note)
        self.assertIn("not a sequence", note)

    def test_prompt_text_includes_multi_view_note_when_multiple_keyframes(
        self,
    ) -> None:
        _, request, _ = load_development_case(DEVELOPMENT_CONFIG, "development-0001")
        request = dict(request)
        request["evidence"] = dict(request["evidence"])
        request["evidence"]["keyframes"] = list(request["evidence"]["keyframes"]) + [
            {"ref": "keyframe-2", "path": "roi.jpg", "sha256": "0" * 64},
        ]
        prompt = _prompt_text(request)
        self.assertIn("SAME instant", prompt)

    def test_allows_motion_claim_when_clip_frames_actually_shown(self) -> None:
        _, request, _ = load_development_case(
            DEVELOPMENT_CONFIG,
            "development-0001",
        )
        assessment = {
            "observations": [
                {"claim_vi": "Xe máy đang di chuyển qua giao lộ."}
            ]
        }
        validate_grounding_policy(assessment, request, clip_frames_shown=True)


if __name__ == "__main__":
    unittest.main()
