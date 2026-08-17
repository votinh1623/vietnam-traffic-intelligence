"""Reproducible Ultralytics detector training with locked-test preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SCRIPTS = PROJECT_ROOT / "scripts" / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from lock_test_split import verify_lock  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def git_command(*args: str) -> tuple[int, str]:
    command = [
        "git",
        "-c",
        f"safe.directory={PROJECT_ROOT.as_posix()}",
        *args,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, (completed.stdout or completed.stderr).strip()


def git_snapshot() -> dict[str, Any]:
    commit_code, commit = git_command("rev-parse", "HEAD")
    status_code, status = git_command("status", "--porcelain")
    return {
        "commit": commit if commit_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status": status.splitlines() if status else [],
    }


def environment_snapshot() -> dict[str, Any]:
    packages = (
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python",
        "numpy",
        "pandas",
        "PyYAML",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"

    gpu = {"cuda_available": False}
    try:
        import torch

        gpu = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as error:  # environment evidence, not training logic
        gpu["error"] = str(error)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "gpu": gpu,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("experiment", "model", "dataset", "train", "provenance"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    return config


def preflight(config_path: Path, smoke: bool = False) -> tuple[dict[str, Any], list[str]]:
    config = load_config(config_path)
    weights = resolve_project_path(config["model"]["weights"])
    data_yaml = resolve_project_path(config["dataset"]["data_yaml"])
    manifest = resolve_project_path(config["dataset"]["materialization_manifest"])
    lock_path = resolve_project_path(config["dataset"]["test_lock"])
    dataset_root = data_yaml.parent
    blockers = []
    for label, path in (
        ("model weights", weights),
        ("data yaml", data_yaml),
        ("materialization manifest", manifest),
        ("test lock", lock_path),
    ):
        if not path.is_file():
            blockers.append(f"missing {label}: {path}")

    lock_verification: dict[str, Any] = {"valid": False, "errors": ["not checked"]}
    if lock_path.is_file() and dataset_root.is_dir():
        lock_verification = verify_lock(dataset_root, lock_path)
        if not lock_verification["valid"]:
            blockers.extend(f"test lock: {error}" for error in lock_verification["errors"])
        expected = config["dataset"].get("test_lock_sha256")
        if expected != lock_verification["computed_lock_sha256"]:
            blockers.append("config test-lock hash does not match verified lock")

    git = git_snapshot()
    if not smoke:
        if config["provenance"].get("require_git_commit") and not git["commit"]:
            blockers.append("repository has no Git commit")
        if config["provenance"].get("require_clean_worktree") and git["dirty"]:
            blockers.append("Git worktree is dirty")

    evidence = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights) if weights.is_file() else None,
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256_file(data_yaml) if data_yaml.is_file() else None,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "test_lock_verification": lock_verification,
        "git": git,
        "environment": environment_snapshot(),
        "smoke": smoke,
    }
    return evidence, blockers


def training_arguments(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    arguments = dict(config["train"])
    arguments["data"] = str(resolve_project_path(config["dataset"]["data_yaml"]))
    arguments["project"] = str(resolve_project_path(arguments["project"]))
    if smoke:
        arguments.update(
            {
                "epochs": 1,
                "imgsz": 640,
                "batch": 2,
                "fraction": 0.10,
                "plots": False,
                "name": f"{arguments['name']}_smoke",
            }
        )
    return arguments


def run_training(config_path: Path, smoke: bool) -> int:
    config = load_config(config_path)
    evidence, blockers = preflight(config_path, smoke=smoke)
    if blockers:
        print(json.dumps({"status": "blocked", "blockers": blockers, "evidence": evidence}, indent=2))
        return 2

    run_id = f"{config['experiment']['id']}{'_smoke' if smoke else ''}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_root = resolve_project_path(config["provenance"]["output_root"]) / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    run_manifest = output_root / "run.json"
    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "evidence": evidence,
        "training_arguments": training_arguments(config, smoke),
    }
    run_manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        from ultralytics import YOLO

        model = YOLO(str(resolve_project_path(config["model"]["weights"])))
        result = model.train(**record["training_arguments"])
        save_dir = Path(result.save_dir).resolve()
        best = save_dir / "weights" / "best.pt"
        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "save_dir": str(save_dir),
                "best_weights_sha256": sha256_file(best) if best.is_file() else None,
            }
        )
    except Exception as error:
        record.update({"status": "failed", "error": repr(error)})
        run_manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise
    run_manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.dry_run:
        evidence, blockers = preflight(config_path, smoke=args.smoke)
        print(
            json.dumps(
                {"status": "ready" if not blockers else "blocked", "blockers": blockers, "evidence": evidence},
                indent=2,
            )
        )
        return 0 if not blockers else 2
    return run_training(config_path, smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())

