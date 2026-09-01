"""Transformers adapter for one frozen development VLM case."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .contracts import ContractError, build_vlm_request, validate_vlm_assessment
from .freeze import file_sha256, verify_evidence_lock


_KEYFRAME_MOTION_PHRASES = (
    "đang di chuyển",
    "đi lên",
    "đi xuống",
    "hướng lên",
    "hướng xuống",
    " is moving",
    " are moving",
)


def validate_grounding_policy(
    assessment: dict[str, Any],
    request: dict[str, Any],
    *,
    clip_frames_shown: bool,
) -> None:
    """Reject motion claims unless the model was actually shown clip frames.

    `clip_frames_shown` must reflect what was actually fed to the model,
    not whether the request references clip evidence -- `run_vlm_case`
    below only ever loads `keyframes[0]`, so a request with clip evidence
    the model never saw must still be treated as keyframe-only.
    """
    if clip_frames_shown:
        return
    for index, observation in enumerate(assessment["observations"]):
        claim = " " + observation["claim_vi"].casefold()
        if any(phrase in claim for phrase in _KEYFRAME_MOTION_PHRASES):
            raise ContractError(
                f"observations[{index}] claims motion from keyframe-only evidence"
            )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one top-level JSON object, allowing an optional Markdown fence."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        trailing = text[index + end :].strip()
        if trailing and trailing not in {"```", "```json"}:
            continue
        return payload
    raise ValueError("VLM output does not contain one valid JSON object")


def load_prompts(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Load the prompt file the config declares under `prompts:`.

    Both reasoning CLIs (run_vlm.py, run_llm.py) must call this instead of
    hardcoding a prompts_*.yaml filename -- a config that still points at
    an older prompt version (e.g. prompts_v1.yaml, which has the fixed
    prompt-copying bug prompts_v3.yaml fixed) should be visible in the
    config file, not buried in the CLI.
    """
    import yaml

    prompts_name = config.get("prompts")
    if not isinstance(prompts_name, str) or not prompts_name:
        raise ValueError("config must declare a `prompts:` file name")
    prompts_path = project_root / "configs" / "reasoning" / prompts_name
    if not prompts_path.is_file():
        raise FileNotFoundError(f"declared prompts file is unavailable: {prompts_path}")
    return yaml.safe_load(prompts_path.read_text(encoding="utf-8"))


def load_development_case(
    config_path: Path, case_id: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("split") != "development":
        raise ValueError("VLM development runner refuses non-development splits")
    lock_path = Path(config["input_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_evidence_lock(lock)
    if lock.get("split") != "development":
        raise ValueError("reasoning lock is not a development split")
    try:
        case = next(item for item in lock["cases"] if item["case_id"] == case_id)
    except StopIteration as error:
        raise ValueError(f"unknown development case: {case_id}") from error
    request = build_vlm_request(case)
    artifact_root = Path(config["artifact_root"])
    for collection in ("keyframes", "clips"):
        for artifact in request["evidence"][collection]:
            path = artifact_root / artifact["path"]
            if not path.is_file():
                raise FileNotFoundError(f"evidence artifact is unavailable: {path}")
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"evidence artifact SHA-256 mismatch: {path}")
    return config, request, artifact_root


def _multi_view_note(request: dict[str, Any]) -> str:
    """Explain multiple images, when present, as same-instant crops -- not
    a sequence over time. Without this, a model shown 2-3 images risks
    inferring motion between them, which validate_grounding_policy would
    then correctly reject since clip_frames_shown is still False for these
    (still images of one moment carry no motion information, unlike real
    clip frames)."""
    keyframes = request["evidence"]["keyframes"]
    if len(keyframes) <= 1:
        return ""
    refs = ", ".join(keyframe["ref"] for keyframe in keyframes)
    return (
        f"\n\n{len(keyframes)} images are provided ({refs}), in this order: "
        "full frame, then (if present) a cropped region of interest, then "
        "(if present) a tight crop around the specific vehicle in this "
        "event. All images are the SAME instant in time, only the field of "
        "view differs -- they are not a sequence and show no motion. Do "
        "not describe movement, direction, or change between them."
    )


def _prompt_text(request: dict[str, Any]) -> str:
    event = request["event"]
    visual_context = {
        field: event[field]
        for field in (
            "schema_version",
            "event_id",
            "event_type",
            "frame_index",
            "timestamp_s",
        )
    }
    # claim_vi below is a schema placeholder, not a model answer: it starts
    # with "<" and reads as an instruction, not prose, specifically so a
    # small model cannot satisfy the request by copying it verbatim. Every
    # run before this fix reproduced the previous literal example
    # ("Quan sát thấy các phương tiện trong khung hình.") word for word --
    # see output/reasoning/adhoc/*.json and output/reasoning/dev_v1/*.json --
    # so the placeholder must not itself be valid-looking Vietnamese prose.
    output_shape = {
        "schema_version": 1,
        "case_id": request["case_id"],
        "event_id": request["event"]["event_id"],
        "observations": [
            {
                "claim_vi": "<mo ta cu the: loai phuong tien chiem da so va cac "
                "loai khac, mat do (thua/vua/dong/rat dong); KHONG duoc chep "
                "nguyen van vi du nay>",
                "confidence": 0.5,
                "evidence_refs": ["keyframe-1"],
            }
        ],
        "incident_assessment": {
            "status": "uncertain",
            "category": "none",
            "confidence": 0.5,
        },
        "limitations": ["..."],
    }
    return (
        "Event identity JSON (not visual ground truth):\n"
        + json.dumps(visual_context, ensure_ascii=False, sort_keys=True)
        + _multi_view_note(request)
        + "\n\nOutput exactly one JSON object matching this shape (claim_vi "
        "below is a placeholder describing what to write, not example text "
        "to copy):\n"
        + json.dumps(output_shape, ensure_ascii=False)
    )


def run_vlm_case(
    *,
    config: dict[str, Any],
    request: dict[str, Any],
    artifact_root: Path,
    model_dir: Path,
    system_prompt: str,
) -> dict[str, Any]:
    """Load the pinned local model, generate once, and validate its JSON."""
    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not model_dir.is_dir():
        raise FileNotFoundError(f"local VLM directory is unavailable: {model_dir}")
    keyframes = request["evidence"]["keyframes"]
    if not keyframes:
        raise ValueError("keyframe-first VLM policy requires a keyframe")
    # All entries in evidence.keyframes are loaded and shown, not just the
    # first -- multi-view evidence (full frame + ROI crop + event crop, see
    # src/vn_traffic/evidence.py) puts additional still-image views of the
    # same instant here rather than in a separate collection. _prompt_text
    # explains this ordering to the model via _multi_view_note.
    images = [
        Image.open(artifact_root / keyframe["path"]).convert("RGB")
        for keyframe in keyframes
    ]
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float16,
        device_map="cuda",
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt.strip()}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} for image in images
            ] + [
                {"type": "text", "text": _prompt_text(request)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    do_sample = config["generation"]["do_sample"]
    base_seed = config["generation"].get("seed", 0)
    # Retrying only helps under sampling: greedy decoding (do_sample=False)
    # is deterministic, so a retry would replay the exact same failure.
    # Measured need for this: real runs against both WP1 demo videos found
    # ~60-67% of greedy-decoded outputs degenerate into diacritic-free
    # Vietnamese that paraphrases the prompt's own instructions instead of
    # describing the image (see prompts_v4.yaml/prompts_v5.yaml notes) --
    # a per-case failure mode, not something a different prompt wording
    # eliminated, so retrying with a fresh sample is the mitigation.
    max_attempts = config["generation"].get("max_attempts", 3) if do_sample else 1

    started = time.perf_counter()
    contract_status = "invalid"
    contract_error = None
    assessment: dict[str, Any] | None = None
    raw_text = ""
    for attempt in range(max_attempts):
        if do_sample:
            torch.manual_seed(base_seed + attempt)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=config["vlm"]["max_new_tokens"],
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_text = processor.decode(generated, skip_special_tokens=True)
        try:
            assessment = extract_json_object(raw_text)
            validate_vlm_assessment(assessment, request)
            # Still False even with multi-view evidence: every entry in
            # evidence.keyframes (full frame, ROI crop, event crop) is a
            # still image of the same instant, not a real clip -- see
            # _multi_view_note. This call path never loads
            # request["evidence"]["clips"] at all, so clip_frames_shown
            # stays False regardless of whether the request references
            # clip evidence.
            validate_grounding_policy(assessment, request, clip_frames_shown=False)
            contract_status = "valid"
            contract_error = None
            break
        except (ValueError, ContractError) as error:
            contract_status = "invalid"
            contract_error = str(error)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    return {
        "schema_version": 1,
        "case_id": request["case_id"],
        "model_id": config["vlm"]["model_id"],
        "revision": config["vlm"]["revision"],
        "dtype": config["vlm"]["dtype"],
        "elapsed_s": elapsed_s,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "attempts_used": attempt + 1,
        "max_attempts": max_attempts,
        "contract_status": contract_status,
        "contract_error": contract_error,
        "raw_text": raw_text,
        "assessment": assessment,
    }
