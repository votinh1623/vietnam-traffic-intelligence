"""Two-reviewer annotation workflow for frozen reasoning evidence sets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .freeze import verify_evidence_lock


ANNOTATION_SCHEMA_VERSION = 1
_ANNOTATION_FIELDS = {
    "schema_version",
    "case_id",
    "reviewer_id",
    "annotation_status",
    "evidence_quality",
    "visible_density",
    "visible_classes",
    "event_visual_support",
    "incident_status",
    "incident_category",
    "observations_vi",
    "reference_summary_vi",
    "required_limitations_vi",
    "notes_vi",
}
_EVIDENCE_QUALITY = {"clear", "partial", "poor"}
_VISIBLE_DENSITY = {"low", "medium", "high", "uncertain"}
_VISIBLE_CLASSES = {"pedestrian", "car", "motorcycle", "bus", "truck", "other"}
_EVENT_VISUAL_SUPPORT = {"supported", "not_supported", "insufficient"}
_INCIDENT_STATUS = {"observed", "not_observed", "uncertain"}
_INCIDENT_CATEGORY = {
    "none",
    "collision",
    "stalled_vehicle",
    "road_hazard",
    "other",
}
_CATEGORICAL_FIELDS = (
    "evidence_quality",
    "visible_density",
    "visible_classes",
    "event_visual_support",
    "incident_status",
    "incident_category",
)


def build_annotation_template(
    lock: dict[str, Any], reviewer_id: str
) -> list[dict[str, Any]]:
    verify_evidence_lock(lock)
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must be non-empty")
    return [
        {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "case_id": case["case_id"],
            "reviewer_id": reviewer_id,
            "annotation_status": "pending",
            "evidence_quality": None,
            "visible_density": None,
            "visible_classes": [],
            "event_visual_support": None,
            "incident_status": None,
            "incident_category": None,
            "observations_vi": [],
            "reference_summary_vi": None,
            "required_limitations_vi": [],
            "notes_vi": "",
        }
        for case in lock["cases"]
    ]


def build_review_index(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a compact case-to-local-media index for human reviewers."""
    verify_evidence_lock(lock)
    run_root = Path("output") / "pipeline" / lock["source"]["run_id"]
    rows = []
    for case in lock["cases"]:
        event = case["event"]
        evidence = case["evidence"]
        keyframe = evidence.get("keyframe")
        clip = evidence.get("clip")
        rows.append(
            {
                "case_id": case["case_id"],
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "frame_index": event["frame_index"],
                "timestamp_s": event["timestamp_s"],
                "keyframe_path": (
                    (run_root / keyframe["path"]).as_posix() if keyframe else ""
                ),
                "clip_path": (run_root / clip["path"]).as_posix() if clip else "",
                "event_json": json.dumps(
                    event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return rows


def _string_list(value: Any, field: str, *, require_nonempty: bool) -> None:
    if not isinstance(value, list) or (require_nonempty and not value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")


def validate_annotation_set(
    records: list[dict[str, Any]],
    lock: dict[str, Any],
    *,
    require_complete: bool,
) -> str:
    """Validate exact case coverage and return the single reviewer ID."""
    verify_evidence_lock(lock)
    if not isinstance(records, list):
        raise ValueError("annotations must be a list")
    expected_ids = [case["case_id"] for case in lock["cases"]]
    actual_ids = [record.get("case_id") for record in records]
    if actual_ids != expected_ids:
        raise ValueError("annotation cases must exactly match lock order and coverage")

    reviewer_ids: set[str] = set()
    for record in records:
        if set(record) != _ANNOTATION_FIELDS:
            raise ValueError(f"annotation fields differ for {record.get('case_id')}")
        if record["schema_version"] != ANNOTATION_SCHEMA_VERSION:
            raise ValueError("unsupported annotation schema version")
        reviewer_id = record["reviewer_id"]
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError("reviewer_id must be non-empty")
        reviewer_ids.add(reviewer_id)
        status = record["annotation_status"]
        if status not in {"pending", "complete"}:
            raise ValueError("annotation_status must be pending or complete")
        if require_complete and status != "complete":
            raise ValueError(f"annotation is still pending: {record['case_id']}")
        if status == "pending":
            continue

        if record["evidence_quality"] not in _EVIDENCE_QUALITY:
            raise ValueError("unsupported evidence_quality")
        if record["visible_density"] not in _VISIBLE_DENSITY:
            raise ValueError("unsupported visible_density")
        visible_classes = record["visible_classes"]
        if not isinstance(visible_classes, list) or len(visible_classes) != len(
            set(visible_classes)
        ):
            raise ValueError("visible_classes must be a unique list")
        if not set(visible_classes) <= _VISIBLE_CLASSES:
            raise ValueError("visible_classes contains an unsupported class")
        if record["event_visual_support"] not in _EVENT_VISUAL_SUPPORT:
            raise ValueError("unsupported event_visual_support")
        if record["incident_status"] not in _INCIDENT_STATUS:
            raise ValueError("unsupported incident_status")
        if record["incident_category"] not in _INCIDENT_CATEGORY:
            raise ValueError("unsupported incident_category")
        if (
            record["incident_status"] == "not_observed"
            and record["incident_category"] != "none"
        ):
            raise ValueError("not_observed incident must use category none")
        _string_list(record["observations_vi"], "observations_vi", require_nonempty=True)
        if (
            not isinstance(record["reference_summary_vi"], str)
            or not record["reference_summary_vi"].strip()
        ):
            raise ValueError("reference_summary_vi must be non-empty")
        _string_list(
            record["required_limitations_vi"],
            "required_limitations_vi",
            require_nonempty=True,
        )
        if not isinstance(record["notes_vi"], str):
            raise ValueError("notes_vi must be text")

    if len(reviewer_ids) != 1:
        raise ValueError("one annotation file must contain exactly one reviewer ID")
    return next(iter(reviewer_ids))


def build_adjudication_queue(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    lock: dict[str, Any],
) -> dict[str, Any]:
    first_reviewer = validate_annotation_set(first, lock, require_complete=True)
    second_reviewer = validate_annotation_set(second, lock, require_complete=True)
    if first_reviewer == second_reviewer:
        raise ValueError("adjudication requires two distinct reviewers")

    cases = []
    disagreement_cases = 0
    for first_record, second_record in zip(first, second):
        disagreements = {
            field: {
                first_reviewer: first_record[field],
                second_reviewer: second_record[field],
            }
            for field in _CATEGORICAL_FIELDS
            if first_record[field] != second_record[field]
        }
        if disagreements:
            disagreement_cases += 1
        cases.append(
            {
                "case_id": first_record["case_id"],
                "categorical_disagreements": disagreements,
                "text_candidates": {
                    first_reviewer: {
                        "observations_vi": first_record["observations_vi"],
                        "reference_summary_vi": first_record["reference_summary_vi"],
                        "required_limitations_vi": first_record[
                            "required_limitations_vi"
                        ],
                        "notes_vi": first_record["notes_vi"],
                    },
                    second_reviewer: {
                        "observations_vi": second_record["observations_vi"],
                        "reference_summary_vi": second_record[
                            "reference_summary_vi"
                        ],
                        "required_limitations_vi": second_record[
                            "required_limitations_vi"
                        ],
                        "notes_vi": second_record["notes_vi"],
                    },
                },
                "adjudication_status": "pending",
                "adjudicated_annotation": None,
            }
        )
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "set_id": lock["set_id"],
        "input_lock_sha256": lock["lock_sha256"],
        "reviewers": [first_reviewer, second_reviewer],
        "case_count": len(cases),
        "categorical_disagreement_cases": disagreement_cases,
        "cases": cases,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_review_index_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "case_id",
        "event_id",
        "event_type",
        "frame_index",
        "timestamp_s",
        "keyframe_path",
        "clip_path",
        "event_json",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
