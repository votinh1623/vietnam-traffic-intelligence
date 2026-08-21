"""Run the frozen VisDrone copy-paste augmentation pilot.

Same structure as train_visdrone_highres.py, adapted for a data.yaml whose
`train` field is a text-file image list (copy-paste augmented images mixed
with the originals) instead of a single directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from train_detector import (
    environment_snapshot,
    git_snapshot,
    resolve_project_path,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"experiment", "model", "dataset", "train", "gates", "provenance"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if config["dataset"]["selection_split"] != "val":
        raise ValueError("pilot selection must use val")
    if config["dataset"]["forbidden_split"] != "test":
        raise ValueError("test must remain the forbidden split")
    return config


def dataset_counts(data_yaml: Path) -> dict[str, int]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    counts = {}
    for split in ("train", "val"):
        target = root / data[split]
        if target.is_dir():
            counts[split] = sum(
                path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                for path in target.iterdir()
                if path.is_file()
            )
        else:
            counts[split] = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def training_arguments(
    config: dict[str, Any], smoke: bool, *, run_name: str | None = None
) -> dict[str, Any]:
    arguments = dict(config["train"])
    arguments["data"] = str(resolve_project_path(config["dataset"]["data_yaml"]))
    arguments["project"] = str(resolve_project_path(arguments["project"]))
    if smoke:
        arguments.update(
            {
                "epochs": 1,
                "fraction": 0.02,
                "plots": False,
                "name": f"{arguments['name']}_smoke",
            }
        )
    if run_name is not None:
        arguments["name"] = run_name
    return arguments


def preflight(config_path: Path, *, smoke: bool) -> tuple[dict[str, Any], list[str]]:
    config = load_config(config_path)
    weights = resolve_project_path(config["model"]["weights"])
    data_yaml = resolve_project_path(config["dataset"]["data_yaml"])
    blockers = []
    if not weights.is_file():
        blockers.append(f"missing model weights: {weights}")
    elif sha256_file(weights) != config["model"]["expected_sha256"]:
        blockers.append("model weights SHA-256 does not match config")
    if not data_yaml.is_file():
        blockers.append(f"missing data YAML: {data_yaml}")
        counts = {}
    else:
        counts = dataset_counts(data_yaml)
        for split in ("train", "val"):
            expected = int(config["dataset"][f"expected_{split}_images"])
            if counts.get(split) != expected:
                blockers.append(
                    f"{split} image count changed: expected {expected}, got {counts.get(split)}"
                )
    git = git_snapshot()
    if not smoke and config["provenance"].get("require_clean_worktree") and git["dirty"]:
        blockers.append("full pilot requires a clean Git worktree")
    evidence = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights) if weights.is_file() else None,
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256_file(data_yaml) if data_yaml.is_file() else None,
        "dataset_image_counts": counts,
        "git": git,
        "environment": environment_snapshot(),
        "smoke": smoke,
    }
    return evidence, blockers


def run_training(config_path: Path, *, smoke: bool) -> int:
    config = load_config(config_path)
    evidence, blockers = preflight(config_path, smoke=smoke)
    if blockers:
        print(json.dumps({"status": "blocked", "blockers": blockers, "evidence": evidence}, indent=2))
        return 2
    run_id = f"{config['experiment']['id']}{'_smoke' if smoke else ''}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    record_root = resolve_project_path(config["provenance"]["output_root"]) / run_id
    record_root.mkdir(parents=True, exist_ok=False)
    manifest = record_root / "run.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "claim_boundary": (
            "Training smoke only; no accuracy claim."
            if smoke
            else "Validation-selected copy-paste augmentation pilot; no test data used during training."
        ),
        "evidence": evidence,
        "training_arguments": training_arguments(config, smoke, run_name=run_id),
        "locked_gates": config["gates"],
    }
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        import torch
        from ultralytics import YOLO

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
                "peak_cuda_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                ),
                "peak_cuda_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as error:
        record.update({"status": "failed", "error": repr(error)})
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    return run_training(resolve_project_path(str(args.config)), smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
