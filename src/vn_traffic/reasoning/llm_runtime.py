"""Transformers adapter for producing a report from validated VLM output."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .contracts import (
    ContractError,
    build_llm_request,
    deterministic_action_level,
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


def _numeric_facts(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"source_path": f"event.measurements.{name}", "value": value}
        for name, value in event.get("measurements", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def assemble_llm_report(
    generated_fields: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Merge generated prose with authoritative fields owned by the pipeline."""
    if set(generated_fields) != {"summary_vi", "action_message_vi"}:
        raise ContractError(
            "LLM prose output must contain only summary_vi and action_message_vi"
        )
    event = request["vlm_request"]["event"]
    report = {
        "schema_version": 1,
        "case_id": request["vlm_request"]["case_id"],
        "event_id": event["event_id"],
        "summary_vi": generated_fields["summary_vi"],
        "traffic_state": event.get("current_state", "UNSPECIFIED"),
        "numeric_facts": _numeric_facts(event),
        "visual_findings": [
            observation["claim_vi"]
            for observation in request["vlm_assessment"]["observations"]
        ],
        "action": {
            "level": deterministic_action_level(request["vlm_assessment"]),
            "message_vi": generated_fields["action_message_vi"],
        },
        "limitations": request["vlm_assessment"]["limitations"],
    }
    validate_llm_report(report, request)
    return report


def build_report_prompt(request: dict[str, Any]) -> str:
    """Ask the LLM only for prose fields; authoritative fields are assembled later."""
    event = request["vlm_request"]["event"]
    if "current_state" in event:
        state_instruction = (
            f"The deterministic traffic_state is {event['current_state']!r}; "
            "state exactly that state and do not rename or reinterpret it. "
        )
    else:
        state_instruction = (
            "This event has no current_state. Do not invent or name a traffic "
            "state (including normal, light, medium, dense, congested, or their "
            "Vietnamese equivalents); describe the event_type instead. "
        )
    return (
        "Input JSON:\n"
        + json.dumps(request, ensure_ascii=False, sort_keys=True)
        + "\n\nReturn exactly one JSON object containing only summary_vi and "
        "action_message_vi. "
        "Write a concrete Vietnamese summary_vi about this specific event using "
        "only the input. "
        + state_instruction
        + "Then mention the dominant vehicle "
        "types and density qualifier if vlm_assessment.observations mentions "
        "them. Do not write a generic sentence like 'quan sat thay cac phuong "
        "tien trong khung hinh' -- if vlm_assessment truly has no specific "
        "detail, say explicitly that visual detail is limited instead of "
        "restating that generic phrase. action_message_vi must be one cautious "
        "Vietnamese recommendation. Do not choose an action level; the "
        "application derives it deterministically from incident_assessment. "
        "Do not output IDs, numeric_facts, "
        "traffic_state, visual_findings, limitations, schema fields, "
        "explanations, or Markdown; the application owns those authoritative "
        "fields."
    )


def load_llm(model_dir: Path) -> tuple[Any, Any]:
    """Load (tokenizer, model) once for reuse across many run_llm_case calls
    -- mirrors vlm_runtime.load_vlm. run_llm_case's default per-call load
    exists for execution_policy: sequential_load_run_unload (crash
    isolation), but reloading the checkpoint from disk on every case in a
    many-case batch (e.g. pipeline_stage.run_reasoning_stage) is unnecessary
    overhead once the run is otherwise stable. Pass the result to
    run_llm_case's model/tokenizer params to reuse it."""
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
    return tokenizer, model


def run_llm_case(
    *,
    config: dict[str, Any],
    request: dict[str, Any],
    model_dir: Path,
    system_prompt: str,
    tokenizer: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Generate once and validate its JSON. Loads the pinned local model
    unless `tokenizer`/`model` are both given (see load_llm) -- pass both
    or neither, never just one."""
    import torch

    if (tokenizer is None) != (model is None):
        raise ValueError("pass both tokenizer and model, or neither")
    if model is None and not model_dir.is_dir():
        raise FileNotFoundError(f"local LLM directory is unavailable: {model_dir}")
    if model is None:
        tokenizer, model = load_llm(model_dir)
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
    generated_fields = extract_json_object(raw_text)
    contract_status = "valid"
    contract_error = None
    report = None
    try:
        report = assemble_llm_report(generated_fields, request)
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
        "generated_fields": generated_fields,
        "report": report,
    }


def prepare_llm_request(
    result_path: Path, vlm_request: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a stored VLM result and construct the downstream request."""
    result = load_validated_vlm_result(result_path, vlm_request)
    return result, build_llm_request(vlm_request, result["assessment"])
