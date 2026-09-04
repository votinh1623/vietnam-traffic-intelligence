"""Wire VLM + LLM reasoning directly into a completed analytics run.

Unlike run_vlm.py/run_llm.py (which replay one frozen development case from
a manifest built by freeze_evidence_set.py), this builds cases straight from
a fresh run's events.jsonl + evidence.jsonl -- no manifest, no freeze step.
It exists so `reasoning.enabled: true` in a pipeline config can turn a
perception+analytics+evidence run into a fully-described one in a single
CLI invocation (see cli.py), rather than requiring a separate freeze +
per-case CLI round trip meant for calibration work.

Only analytics events or visual-scan routing triggers whose type is in
ReasoningConfig.event_types AND that have an evidence record are described --
see ReasoningConfig's docstring for why
line_crossing is excluded by default.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import yaml

from .contracts import build_llm_request, build_vlm_request
from .llm_runtime import load_llm, run_llm_case
from .vlm_runtime import load_vlm, run_vlm_case

if TYPE_CHECKING:
    from ..config import ReasoningConfig


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_cases(run_dir: Path, event_types: tuple[str, ...]) -> list[dict[str, Any]]:
    """One case per (event, evidence) pair whose event_type qualifies.

    Reuses events.jsonl's own record as case["event"] verbatim -- contracts.
    _event() only requires event_id/event_type/frame_index/timestamp_s to be
    present and valid, and passes the rest through untouched, so track_id/
    class_name/measurements/current_state all reach the VLM/LLM prompts for
    free without a second schema to keep in sync.
    """
    events = {event["event_id"]: event for event in _load_jsonl(run_dir / "events.jsonl")}
    evidence_records = _load_jsonl(run_dir / "evidence.jsonl")
    cases: list[dict[str, Any]] = []
    for record in evidence_records:
        event = events.get(record["event_id"])
        if event is None or event["event_type"] not in event_types:
            continue
        media: dict[str, Any] = {}
        for artifact_name in ("keyframe", "roi_crop", "event_crop", "clip"):
            artifact = record.get(artifact_name)
            if artifact is not None:
                media[artifact_name] = {"path": artifact["path"], "sha256": artifact["sha256"]}
        if not media:
            continue
        cases.append(
            {
                "case_id": f"{run_dir.name}-{event['event_id']}",
                "event": event,
                "evidence": media,
            }
        )
    return cases


def run_reasoning_stage(
    *, reasoning_config: "ReasoningConfig", run_dir: Path, project_root: Path,
) -> Path:
    """Describe every qualifying event in run_dir, writing run_dir/reasoning.jsonl.

    One record per event: {event_id, event_type, vlm, llm}. vlm/llm carry
    the same contract_status/assessment(report) shape run_vlm.py/run_llm.py
    write to disk, so downstream consumers (dashboard, audits) do not need a
    second parser for the pipeline-integrated path vs. the manifest path.

    Model loading follows execution_policy: sequential_load_run_unload
    (see docs/quickstart.md) -- the VLM is loaded once and reused across all
    cases (run_vlm_case's model/processor reuse params), then freed before
    the LLM loads, since both loaded together would exceed the 6 GB budget
    this project targets.
    """
    output_path = run_dir / "reasoning.jsonl"
    cases = build_cases(run_dir, reasoning_config.event_types)
    total = len(cases)
    print(f"[reasoning] {total} case(s) qualify from {run_dir.name}", flush=True)
    if not cases:
        output_path.write_text("", encoding="utf-8")
        return output_path

    prompts_path = project_root / "configs" / "reasoning" / reasoning_config.prompts
    if not prompts_path.is_file():
        raise FileNotFoundError(f"declared prompts file is unavailable: {prompts_path}")
    prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))

    vlm_config = {"vlm": reasoning_config.vlm, "generation": reasoning_config.generation}
    llm_config = {"llm": reasoning_config.llm, "generation": reasoning_config.generation}

    vlm_by_case: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    print("[reasoning] loading VLM...", flush=True)
    load_started = time.perf_counter()
    processor, model = load_vlm(reasoning_config.vlm_model_dir)
    print(f"[reasoning] VLM loaded in {time.perf_counter() - load_started:.1f}s", flush=True)
    try:
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            request = build_vlm_request(case)
            result = run_vlm_case(
                config=vlm_config,
                request=request,
                artifact_root=run_dir,
                model_dir=reasoning_config.vlm_model_dir,
                system_prompt=prompts["vlm"]["system"],
                processor=processor,
                model=model,
            )
            vlm_by_case[case["case_id"]] = (request, result)
            print(
                f"[reasoning] VLM {index}/{total} {case['case_id']} "
                f"status={result['contract_status']} "
                f"attempts={result['attempts_used']}/{result['max_attempts']} "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        del model, processor
        import torch

        torch.cuda.empty_cache()

    valid_cases = [case for case in cases if vlm_by_case[case["case_id"]][1]["contract_status"] == "valid"]
    records: list[dict[str, Any]] = []
    llm_model = None
    llm_tokenizer = None
    if valid_cases:
        print("[reasoning] loading LLM...", flush=True)
        load_started = time.perf_counter()
        llm_tokenizer, llm_model = load_llm(reasoning_config.llm_model_dir)
        print(f"[reasoning] LLM loaded in {time.perf_counter() - load_started:.1f}s", flush=True)
    try:
        for index, case in enumerate(cases, start=1):
            vlm_request, vlm_result = vlm_by_case[case["case_id"]]
            record: dict[str, Any] = {
                "case_id": case["case_id"],
                "event_id": case["event"]["event_id"],
                "event_type": case["event"]["event_type"],
                "vlm": vlm_result,
                "llm": None,
            }
            if vlm_result["contract_status"] == "valid":
                started = time.perf_counter()
                llm_request = build_llm_request(vlm_request, vlm_result["assessment"])
                llm_result = run_llm_case(
                    config=llm_config,
                    request=llm_request,
                    model_dir=reasoning_config.llm_model_dir,
                    system_prompt=prompts["llm"]["system"],
                    tokenizer=llm_tokenizer,
                    model=llm_model,
                )
                record["llm"] = llm_result
                print(
                    f"[reasoning] LLM {index}/{total} {case['case_id']} "
                    f"status={llm_result['contract_status']} "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            records.append(record)
    finally:
        if llm_model is not None:
            del llm_model, llm_tokenizer
            import torch

            torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
