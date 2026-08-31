"""Strict versioned contracts between evidence, VLM, and LLM stages."""

from __future__ import annotations

import math
import re
from typing import Any


REASONING_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INCIDENT_STATUSES = {"observed", "not_observed", "uncertain"}
_INCIDENT_CATEGORIES = {
    "none",
    "collision",
    "stalled_vehicle",
    "road_hazard",
    "other",
}
_TRAFFIC_STATES = {"NORMAL", "DENSE", "CONGESTED", "UNSPECIFIED", "UNKNOWN"}
_ACTION_LEVELS = {"none", "monitor", "review", "alert"}


class ContractError(ValueError):
    """Raised when a reasoning payload violates its declared contract."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _exact_keys(
    payload: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    optional = optional or set()
    missing = required - payload.keys()
    extra = payload.keys() - required - optional
    if missing:
        raise ContractError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"{name} has unsupported fields: {sorted(extra)}")


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ContractError(f"{name} must be a safe identifier")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be non-empty text")
    return value


def _confidence(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ContractError(f"{name} must be finite and between 0 and 1")
    return number


def _string_list(value: Any, name: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ContractError(f"{name} must be a list of strings")
    for index, item in enumerate(value):
        _nonempty_text(item, f"{name}[{index}]")
    return value


def _event(payload: Any, name: str = "event") -> dict[str, Any]:
    event = _mapping(payload, name)
    _safe_id(event.get("event_id"), f"{name}.event_id")
    _nonempty_text(event.get("event_type"), f"{name}.event_type")
    if not isinstance(event.get("frame_index"), int) or event["frame_index"] < 0:
        raise ContractError(f"{name}.frame_index must be a non-negative integer")
    timestamp = event.get("timestamp_s")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ContractError(f"{name}.timestamp_s must be numeric")
    if not math.isfinite(float(timestamp)) or timestamp < 0:
        raise ContractError(f"{name}.timestamp_s must be finite and non-negative")
    return event


def _evidence_refs(request: dict[str, Any]) -> set[str]:
    evidence = request["evidence"]
    _exact_keys(
        evidence,
        required={"keyframes", "clips"},
        name="evidence",
    )
    refs: set[str] = set()
    for collection in ("keyframes", "clips"):
        items = evidence.get(collection)
        if not isinstance(items, list):
            raise ContractError(f"evidence.{collection} must be a list")
        for index, item_value in enumerate(items):
            item = _mapping(item_value, f"evidence.{collection}[{index}]")
            _exact_keys(
                item,
                required={"ref", "path", "sha256"},
                name=f"evidence.{collection}[{index}]",
            )
            ref = _safe_id(item.get("ref"), f"evidence.{collection}[{index}].ref")
            if ref in refs:
                raise ContractError(f"duplicate evidence ref: {ref}")
            _nonempty_text(item.get("path"), f"evidence.{collection}[{index}].path")
            sha256 = item.get("sha256")
            if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ContractError(
                    f"evidence.{collection}[{index}].sha256 must be lowercase SHA-256"
                )
            refs.add(ref)
    if not refs:
        raise ContractError("evidence must contain at least one media artifact")
    return refs


def validate_vlm_request(payload: Any) -> None:
    """Validate the frozen request boundary presented to a VLM."""
    request = _mapping(payload, "vlm_request")
    _exact_keys(
        request,
        required={
            "schema_version",
            "case_id",
            "task",
            "locale",
            "event",
            "evidence",
            "constraints",
        },
        name="vlm_request",
    )
    if request["schema_version"] != REASONING_SCHEMA_VERSION:
        raise ContractError("unsupported VLM request schema_version")
    _safe_id(request["case_id"], "case_id")
    if request["task"] != "traffic_event_visual_review":
        raise ContractError("unsupported VLM task")
    if request["locale"] != "vi-VN":
        raise ContractError("locale must be vi-VN")
    _event(request["event"])
    _mapping(request["evidence"], "evidence")
    _evidence_refs(request)
    constraints = _mapping(request["constraints"], "constraints")
    _exact_keys(
        constraints,
        required={
            "deterministic_fields_are_authoritative",
            "infer_physical_speed",
            "infer_event_cause",
        },
        name="constraints",
    )
    if constraints != {
        "deterministic_fields_are_authoritative": True,
        "infer_physical_speed": False,
        "infer_event_cause": False,
    }:
        raise ContractError("VLM safety constraints must not be relaxed")


def build_vlm_request(case: dict[str, Any]) -> dict[str, Any]:
    """Build the exact VLM boundary from one frozen evidence-lock case.

    Multi-view still images (full-frame keyframe, ROI crop, event crop --
    see src/vn_traffic/evidence.py) are all exposed as ordinary entries in
    `evidence.keyframes`, not a new collection: they are all still images
    of the *same instant*, differing only in field of view, so the existing
    keyframes list already fits without a schema change. This is
    deliberately unlike `clips`, which carries real motion over time and is
    gated by validate_grounding_policy's clip_frames_shown.
    """
    evidence_record = _mapping(case.get("evidence"), "case.evidence")
    media: dict[str, list[dict[str, Any]]] = {"keyframes": [], "clips": []}
    keyframe_count = 0
    for artifact_name in ("keyframe", "roi_crop", "event_crop"):
        artifact = evidence_record.get(artifact_name)
        if artifact is not None:
            keyframe_count += 1
            media["keyframes"].append(
                {
                    "ref": f"keyframe-{keyframe_count}",
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                }
            )
    clip_artifact = evidence_record.get("clip")
    if clip_artifact is not None:
        media["clips"].append(
            {
                "ref": "clip-1",
                "path": clip_artifact["path"],
                "sha256": clip_artifact["sha256"],
            }
        )
    request = {
        "schema_version": REASONING_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "task": "traffic_event_visual_review",
        "locale": "vi-VN",
        "event": case["event"],
        "evidence": media,
        "constraints": {
            "deterministic_fields_are_authoritative": True,
            "infer_physical_speed": False,
            "infer_event_cause": False,
        },
    }
    validate_vlm_request(request)
    return request


def validate_vlm_assessment(payload: Any, request_payload: Any) -> None:
    """Validate visual-only model output and its evidence citations."""
    validate_vlm_request(request_payload)
    request = request_payload
    assessment = _mapping(payload, "vlm_assessment")
    _exact_keys(
        assessment,
        required={
            "schema_version",
            "case_id",
            "event_id",
            "observations",
            "incident_assessment",
            "limitations",
        },
        name="vlm_assessment",
    )
    if assessment["schema_version"] != REASONING_SCHEMA_VERSION:
        raise ContractError("unsupported VLM assessment schema_version")
    if assessment["case_id"] != request["case_id"]:
        raise ContractError("VLM assessment case_id does not match request")
    if assessment["event_id"] != request["event"]["event_id"]:
        raise ContractError("VLM assessment event_id does not match request")
    valid_refs = _evidence_refs(request)
    observations = assessment["observations"]
    if not isinstance(observations, list):
        raise ContractError("observations must be a list")
    for index, observation_value in enumerate(observations):
        observation = _mapping(observation_value, f"observations[{index}]")
        _exact_keys(
            observation,
            required={"claim_vi", "confidence", "evidence_refs"},
            name=f"observations[{index}]",
        )
        _nonempty_text(observation["claim_vi"], f"observations[{index}].claim_vi")
        confidence = _confidence(
            observation["confidence"], f"observations[{index}].confidence"
        )
        if confidence == 0:
            raise ContractError(
                f"observations[{index}].confidence must be greater than zero"
            )
        refs = _string_list(
            observation["evidence_refs"],
            f"observations[{index}].evidence_refs",
            allow_empty=False,
        )
        if not set(refs) <= valid_refs:
            raise ContractError(f"observations[{index}] cites unknown evidence")
    incident = _mapping(assessment["incident_assessment"], "incident_assessment")
    _exact_keys(
        incident,
        required={"status", "category", "confidence"},
        name="incident_assessment",
    )
    if incident["status"] not in _INCIDENT_STATUSES:
        raise ContractError("unsupported incident status")
    if incident["category"] not in _INCIDENT_CATEGORIES:
        raise ContractError("unsupported incident category")
    if incident["status"] == "not_observed" and incident["category"] != "none":
        raise ContractError("not_observed incident must use category none")
    _confidence(incident["confidence"], "incident_assessment.confidence")
    _string_list(assessment["limitations"], "limitations")


def build_llm_request(
    vlm_request: dict[str, Any], vlm_assessment: dict[str, Any]
) -> dict[str, Any]:
    """Build an LLM request containing a validated visual assessment."""
    validate_vlm_assessment(vlm_assessment, vlm_request)
    request = {
        "schema_version": REASONING_SCHEMA_VERSION,
        "task": "traffic_event_report",
        "locale": "vi-VN",
        "vlm_request": vlm_request,
        "vlm_assessment": vlm_assessment,
    }
    validate_llm_request(request)
    return request


def validate_llm_request(payload: Any) -> None:
    request = _mapping(payload, "llm_request")
    _exact_keys(
        request,
        required={
            "schema_version",
            "task",
            "locale",
            "vlm_request",
            "vlm_assessment",
        },
        name="llm_request",
    )
    if request["schema_version"] != REASONING_SCHEMA_VERSION:
        raise ContractError("unsupported LLM request schema_version")
    if request["task"] != "traffic_event_report":
        raise ContractError("unsupported LLM task")
    if request["locale"] != "vi-VN":
        raise ContractError("locale must be vi-VN")
    validate_vlm_assessment(request["vlm_assessment"], request["vlm_request"])


def _resolve_numeric_source(event: dict[str, Any], source_path: str) -> float | int:
    if not source_path.startswith("event."):
        raise ContractError("numeric fact source_path must start with event.")
    value: Any = {"event": event}
    for part in source_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"numeric fact source does not exist: {source_path}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"numeric fact source is not numeric: {source_path}")
    if not math.isfinite(float(value)):
        raise ContractError(f"numeric fact source is not finite: {source_path}")
    return value


def validate_llm_report(payload: Any, request_payload: Any) -> None:
    """Validate a final report and enforce exact deterministic numeric facts."""
    validate_llm_request(request_payload)
    request = request_payload
    vlm_request = request["vlm_request"]
    report = _mapping(payload, "llm_report")
    _exact_keys(
        report,
        required={
            "schema_version",
            "case_id",
            "event_id",
            "summary_vi",
            "traffic_state",
            "numeric_facts",
            "visual_findings",
            "action",
            "limitations",
        },
        name="llm_report",
    )
    if report["schema_version"] != REASONING_SCHEMA_VERSION:
        raise ContractError("unsupported LLM report schema_version")
    if report["case_id"] != vlm_request["case_id"]:
        raise ContractError("LLM report case_id does not match request")
    if report["event_id"] != vlm_request["event"]["event_id"]:
        raise ContractError("LLM report event_id does not match request")
    _nonempty_text(report["summary_vi"], "summary_vi")
    if report["traffic_state"] not in _TRAFFIC_STATES:
        raise ContractError("unsupported traffic_state")
    expected_state = vlm_request["event"].get("current_state", "UNSPECIFIED")
    if report["traffic_state"] != expected_state:
        raise ContractError("traffic_state must match the deterministic event")

    numeric_facts = report["numeric_facts"]
    if not isinstance(numeric_facts, list):
        raise ContractError("numeric_facts must be a list")
    seen_sources: set[str] = set()
    for index, fact_value in enumerate(numeric_facts):
        fact = _mapping(fact_value, f"numeric_facts[{index}]")
        _exact_keys(
            fact,
            required={"source_path", "value"},
            name=f"numeric_facts[{index}]",
        )
        source_path = _nonempty_text(
            fact["source_path"], f"numeric_facts[{index}].source_path"
        )
        if source_path in seen_sources:
            raise ContractError(f"duplicate numeric fact source: {source_path}")
        seen_sources.add(source_path)
        expected = _resolve_numeric_source(vlm_request["event"], source_path)
        value = fact["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"numeric_facts[{index}].value must be numeric")
        if float(value) != float(expected):
            raise ContractError(
                f"numeric fact differs from deterministic source: {source_path}"
            )

    _string_list(report["visual_findings"], "visual_findings")
    expected_findings = [
        observation["claim_vi"]
        for observation in request["vlm_assessment"]["observations"]
    ]
    if report["visual_findings"] != expected_findings:
        raise ContractError(
            "visual_findings must exactly preserve validated VLM observations"
        )
    action = _mapping(report["action"], "action")
    _exact_keys(
        action,
        required={"level", "message_vi"},
        name="action",
    )
    if action["level"] not in _ACTION_LEVELS:
        raise ContractError("unsupported action level")
    _nonempty_text(action["message_vi"], "action.message_vi")
    _string_list(report["limitations"], "limitations")
    if report["limitations"] != request["vlm_assessment"]["limitations"]:
        raise ContractError(
            "limitations must exactly preserve validated VLM limitations"
        )
