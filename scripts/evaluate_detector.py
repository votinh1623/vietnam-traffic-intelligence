"""Evaluate the frozen detector once on the content-addressed locked test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from lock_test_split import verify_lock  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *args],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

    status = run("status", "--porcelain").splitlines()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_final_test":
        raise ValueError("detector evaluation config must be frozen_final_test")
    if config.get("dataset", {}).get("split") != "test":
        raise ValueError("detector locked evaluation requires split: test")
    if not config.get("provenance", {}).get("require_clean_worktree"):
        raise ValueError("locked evaluation requires a clean worktree")


def preflight(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    weights = resolve_path(config["model"]["weights"])
    data_yaml = resolve_path(config["dataset"]["data_yaml"])
    manifest = resolve_path(config["dataset"]["materialization_manifest"])
    lock_path = resolve_path(config["dataset"]["test_lock"])
    blockers = []
    for label, path in (
        ("weights", weights),
        ("data YAML", data_yaml),
        ("materialization manifest", manifest),
        ("test lock", lock_path),
    ):
        if not path.is_file():
            blockers.append(f"missing {label}: {path}")
    if weights.is_file() and file_sha256(weights) != config["model"]["expected_sha256"]:
        blockers.append("model SHA-256 mismatch")
    lock_result: dict[str, Any] = {"valid": False, "errors": ["not checked"]}
    if data_yaml.is_file() and lock_path.is_file():
        lock_result = verify_lock(data_yaml.parent, lock_path)
        if not lock_result["valid"]:
            blockers.extend(f"test lock: {error}" for error in lock_result["errors"])
        if (
            lock_result["computed_lock_sha256"]
            != config["dataset"]["expected_test_lock_sha256"]
        ):
            blockers.append("verified test-lock SHA-256 differs from config")
    git = git_snapshot()
    if git["dirty"]:
        blockers.append("Git worktree must be clean for locked-test evaluation")
    evidence = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "model": {
            "path": config["model"]["weights"],
            "sha256": file_sha256(weights) if weights.is_file() else None,
        },
        "dataset": {
            "data_yaml": config["dataset"]["data_yaml"],
            "data_yaml_sha256": file_sha256(data_yaml) if data_yaml.is_file() else None,
            "manifest": config["dataset"]["materialization_manifest"],
            "manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
            "test_lock": config["dataset"]["test_lock"],
            "verification": lock_result,
        },
        "git": git,
    }
    return evidence, blockers


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "ultralytics", "opencv-python", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch

    return {
        "python": sys.version,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
    }


def json_scalars(values: dict[str, Any]) -> dict[str, float | int | None]:
    result = {}
    for name, value in values.items():
        if value is None:
            result[name] = None
        elif hasattr(value, "item"):
            result[name] = value.item()
        else:
            result[name] = value
    return result


def run_evaluation(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence, blockers = preflight(config_path)
    if blockers:
        print(json.dumps({"status": "blocked", "blockers": blockers, "evidence": evidence}, indent=2))
        return 2
    output_root = resolve_path(config["provenance"]["output_root"])
    if output_root.exists():
        raise FileExistsError(f"locked-test output already exists: {output_root}")
    output_root.mkdir(parents=True)
    run_path = output_root / "run.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": config["evaluation_id"],
        "status": "running",
        "claim_boundary": config["provenance"]["claim_boundary"],
        "locked_test_consumed": True,
        "evidence": evidence,
        "parameters": config["validation"],
    }
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        from ultralytics import YOLO

        model = YOLO(str(resolve_path(config["model"]["weights"])))
        metrics = model.val(
            data=str(resolve_path(config["dataset"]["data_yaml"])),
            split="test",
            imgsz=int(config["validation"]["imgsz"]),
            batch=int(config["validation"]["batch"]),
            device=str(config["validation"]["device"]),
            workers=int(config["validation"]["workers"]),
            plots=bool(config["validation"]["plots"]),
            project=str(output_root),
            name="artifacts",
            exist_ok=False,
        )
        class_names = [model.names[index] for index in sorted(model.names)]
        per_class_map50_95 = {
            name: float(metrics.box.maps[index])
            for index, name in enumerate(class_names)
        }
        record.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": json_scalars(metrics.results_dict),
                "per_class_map50_95": per_class_map50_95,
                "speed_ms_per_image": json_scalars(metrics.speed),
                "artifact_directory": str(Path(metrics.save_dir).relative_to(output_root)),
                "environment": environment_snapshot(),
            }
        )
    except Exception as error:
        record.update({"status": "failed", "error": repr(error)})
        run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    if args.dry_run:
        evidence, blockers = preflight(config_path)
        print(json.dumps({"status": "ready" if not blockers else "blocked", "blockers": blockers, "evidence": evidence}, indent=2))
        return 0 if not blockers else 2
    return run_evaluation(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
