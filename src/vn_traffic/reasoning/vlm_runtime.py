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
    assessment: dict[str, Any], request: dict[str, Any]
) -> None:
    """Reject motion claims when the model received only a still keyframe."""
    if request["evidence"]["clips"]:
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
    output_shape = {
        "schema_version": 1,
        "case_id": request["case_id"],
        "event_id": request["event"]["event_id"],
        "observations": [
            {
                "claim_vi": "Quan sát thấy các phương tiện trong khung hình.",
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
        + "\n\nOutput exactly one JSON object matching this shape:\n"
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
    image_path = artifact_root / keyframes[0]["path"]
    image = Image.open(image_path).convert("RGB")
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
                {"type": "image", "image": image},
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
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=config["generation"]["do_sample"],
            max_new_tokens=config["vlm"]["max_new_tokens"],
        )
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = processor.decode(generated, skip_special_tokens=True)
    assessment = extract_json_object(raw_text)
    contract_status = "valid"
    contract_error = None
    try:
        validate_vlm_assessment(assessment, request)
        validate_grounding_policy(assessment, request)
    except ContractError as error:
        contract_status = "invalid"
        contract_error = str(error)
    return {
        "schema_version": 1,
        "case_id": request["case_id"],
        "model_id": config["vlm"]["model_id"],
        "revision": config["vlm"]["revision"],
        "dtype": config["vlm"]["dtype"],
        "elapsed_s": elapsed_s,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "contract_status": contract_status,
        "contract_error": contract_error,
        "raw_text": raw_text,
        "assessment": assessment,
    }
