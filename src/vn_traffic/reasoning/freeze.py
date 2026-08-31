"""Create and verify content-addressed reasoning-input locks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..evidence import EVIDENCE_SCHEMA_VERSION


EVIDENCE_SET_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_evidence_lock(
    *,
    run_dir: Path,
    set_id: str,
    split: str,
) -> dict[str, Any]:
    """Freeze every evidence record in one completed pipeline run."""
    metadata_path = run_dir / "run.json"
    events_path = run_dir / "events.jsonl"
    evidence_path = run_dir / "evidence.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("reasoning evidence can only be frozen from a completed run")
    if metadata.get("evidence", {}).get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "reasoning evidence lock requires evidence schema version "
            f"{EVIDENCE_SCHEMA_VERSION}"
        )

    events = _read_jsonl(events_path)
    evidence_records = _read_jsonl(evidence_path)
    events_by_id = {event.get("event_id"): event for event in events}
    if len(events_by_id) != len(events):
        raise ValueError("events.jsonl contains duplicate event IDs")

    source_path = Path(metadata["source"])
    declared_source_sha256 = metadata["evidence"]["source_video_sha256"]
    if not source_path.is_file():
        raise ValueError(f"source video is unavailable: {source_path}")
    if file_sha256(source_path) != declared_source_sha256:
        raise ValueError("source video SHA-256 differs from evidence manifest")

    resolved_run_dir = run_dir.resolve()
    cases: list[dict[str, Any]] = []
    for index, evidence in enumerate(evidence_records, 1):
        event_id = evidence.get("event_id")
        if event_id not in events_by_id:
            raise ValueError(f"evidence references unknown event: {event_id}")
        event = events_by_id[event_id]
        if event.get("frame_index") != evidence.get("source_frame_index"):
            raise ValueError(f"frame mismatch for event: {event_id}")
        if event.get("event_type") != evidence.get("event_type"):
            raise ValueError(f"event type mismatch for event: {event_id}")
        if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema for event: {event_id}")
        if evidence.get("source_video_sha256") != declared_source_sha256:
            raise ValueError(f"source SHA-256 mismatch for event: {event_id}")
        for artifact_name in ("keyframe", "clip"):
            artifact = evidence.get(artifact_name)
            if artifact is None:
                continue
            relative_path = Path(artifact.get("path", ""))
            artifact_path = (run_dir / relative_path).resolve()
            try:
                artifact_path.relative_to(resolved_run_dir)
            except ValueError as error:
                raise ValueError(
                    f"artifact escapes run directory for event: {event_id}"
                ) from error
            if not artifact_path.is_file():
                raise ValueError(f"missing artifact for event: {event_id}")
            if file_sha256(artifact_path) != artifact.get("sha256"):
                raise ValueError(f"artifact SHA-256 mismatch for event: {event_id}")
        cases.append(
            {
                "case_id": f"{split}-{index:04d}",
                "event_sha256": canonical_sha256(event),
                "evidence_sha256": canonical_sha256(evidence),
                "event": event,
                "evidence": evidence,
            }
        )

    lock: dict[str, Any] = {
        "schema_version": EVIDENCE_SET_SCHEMA_VERSION,
        "reasoning_contract_schema_version": 1,
        "set_id": set_id,
        "split": split,
        "status": "inputs_frozen_annotations_pending",
        "selection_policy": "all evidence records from the declared run",
        "artifact_policy": (
            "media stays local; paths resolve under the source run and hashes "
            "are authoritative"
        ),
        "source": {
            "run_id": metadata["run_id"],
            "video_name": source_path.name,
            "source_video_sha256": declared_source_sha256,
            "frames_processed": metadata["frames_processed"],
            "events_jsonl_sha256": file_sha256(events_path),
            "evidence_jsonl_sha256": file_sha256(evidence_path),
        },
        "case_count": len(cases),
        "cases": cases,
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    return lock


def verify_evidence_lock(payload: dict[str, Any]) -> None:
    """Verify the top-level lock and every embedded case hash."""
    if payload.get("schema_version") != EVIDENCE_SET_SCHEMA_VERSION:
        raise ValueError("unsupported evidence-set schema version")
    expected_lock = payload.get("lock_sha256")
    if not isinstance(expected_lock, str):
        raise ValueError("evidence lock is missing lock_sha256")
    unhashed = dict(payload)
    del unhashed["lock_sha256"]
    if canonical_sha256(unhashed) != expected_lock:
        raise ValueError("evidence-set lock SHA-256 mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list) or payload.get("case_count") != len(cases):
        raise ValueError("evidence-set case count mismatch")
    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in case_ids:
            raise ValueError("evidence-set case IDs must be unique strings")
        case_ids.add(case_id)
        if canonical_sha256(case.get("event")) != case.get("event_sha256"):
            raise ValueError(f"event SHA-256 mismatch for {case_id}")
        if canonical_sha256(case.get("evidence")) != case.get("evidence_sha256"):
            raise ValueError(f"evidence SHA-256 mismatch for {case_id}")


def write_evidence_lock(path: Path, payload: dict[str, Any]) -> None:
    verify_evidence_lock(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
