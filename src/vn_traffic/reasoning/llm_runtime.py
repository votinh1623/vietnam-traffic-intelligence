"""Transformers adapter for producing a report from validated VLM output."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .contracts import (
    ContractError,
    build_llm_request,
    validate_llm_report,
    validate_vlm_assessment,
)
from .vlm_runtime import extract_json_object


def load_validated_vlm_result(
    result_path: Path, vlm_request: dict[str, Any]
) -> dict[str, Any]:
    """Load an audited VLM result and reject mismatched or invalid artifacts."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("contract_status") != "valid":
        raise ContractError("LLM runner requires contract_status=valid VLM output")
    if result.get("case_id") != vlm_request["case_id"]:
        raise ContractError("VLM result case_id does not match the frozen request")
    assessment = result.get("assessment")
    validate_vlm_assessment(assessment, vlm_request)
    return result


def build_report_prompt(request: dict[str, Any]) -> str:
    """Build the user prompt while keeping deterministic and visual facts separate."""
    event = request["vlm_request"]["event"]
    numeric_facts = [
        {"source_path": f"event.measurements.{name}", "value": value}
        for name, value in event.get("measurements", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    output_shape = {
        "schema_version": 1,
        "case_id": request["vlm_request"]["case_id"],
        "event_id": event["event_id"],
        "summary_vi": "Tóm tắt thận trọng dựa trên các đầu vào được cung cấp.",
        "traffic_state": event.get("current_state", "UNSPECIFIED"),
        "numeric_facts": numeric_facts,
        "visual_findings": ["Chỉ diễn đạt lại quan sát có trong VLM assessment."],
        "action": {
            "level": "monitor",
            "message_vi": "Tiếp tục theo dõi và xác minh khi cần.",
        },
        "limitations": ["Nêu giới hạn của evidence và mô hình."],
    }
    return (
        "Input JSON:\n"
        + json.dumps(request, ensure_ascii=False, sort_keys=True)
        + "\n\nReturn exactly one JSON object matching this shape. Preserve every "
        "provided case ID, event ID, traffic state, numeric source_path and numeric "
        "value exactly; do not add numeric facts:\n"
        + json.dumps(output_shape, ensure_ascii=False)
    )


def run_llm_case(
    *,
    config: dict[str, Any],
    request: dict[str, Any],
    model_dir: Path,
    system_prompt: str,
) -> dict[str, Any]:
    """Load the pinned local language model, generate once, and validate JSON."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_dir.is_dir():
        raise FileNotFoundError(f"local LLM directory is unavailable: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float16,
        device_map="cuda",
    )
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": build_report_prompt(request)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=config["llm"]["thinking"],
    ).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=config["generation"]["do_sample"],
            max_new_tokens=config["llm"]["max_new_tokens"],
        )
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    report = extract_json_object(raw_text)
    contract_status = "valid"
    contract_error = None
    try:
        validate_llm_report(report, request)
    except ContractError as error:
        contract_status = "invalid"
        contract_error = str(error)
    return {
        "schema_version": 1,
        "case_id": request["vlm_request"]["case_id"],
        "model_id": config["llm"]["model_id"],
        "revision": config["llm"]["revision"],
        "dtype": config["llm"]["dtype"],
        "elapsed_s": elapsed_s,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "contract_status": contract_status,
        "contract_error": contract_error,
        "raw_text": raw_text,
        "report": report,
    }


def prepare_llm_request(
    result_path: Path, vlm_request: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a stored VLM result and construct the downstream request."""
    result = load_validated_vlm_result(result_path, vlm_request)
    return result, build_llm_request(vlm_request, result["assessment"])
