"""Cache post-NMS low-confidence YOLO proposals for the frozen TVLR oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import cv2
import pandas as pd
import yaml

from benchmark_tracking import (
    file_manifest_sha256,
    file_sha256,
    resolve_path,
    sequence_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_COLUMNS = [
    "sequence",
    "frame",
    "proposal_id",
    "class_name",
    "confidence",
    "x",
    "y",
    "w",
    "h",
]


def git_evidence() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *args],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

    status = run("status", "--porcelain").splitlines()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def export_sequence(
    model: Any,
    *,
    sequence: str,
    frame_paths: list[Path],
    settings: dict[str, Any],
    allowed_classes: set[str],
) -> tuple[pd.DataFrame, list[int]]:
    """Export model boxes without tracking and report max-det saturation frames."""
    records: list[dict[str, Any]] = []
    saturated_frames: list[int] = []
    max_det = int(settings["max_det"])
    for position, frame_path in enumerate(frame_paths, start=1):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"cannot decode frame: {frame_path}")
        result = model.predict(
            image,
            imgsz=int(settings["imgsz"]),
            conf=float(settings["confidence"]),
            iou=float(settings["iou"]),
            max_det=max_det,
            device=str(settings["device"]),
            verbose=False,
        )[0]
        boxes = result.boxes
        frame_index = int(frame_path.stem)
        box_count = 0 if boxes is None else len(boxes)
        if box_count >= max_det:
            saturated_frames.append(frame_index)
        if boxes is not None:
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            proposal_id = 0
            for class_id, confidence, box in zip(classes, confidences, coordinates):
                class_name = str(model.names[class_id])
                if class_name not in allowed_classes:
                    continue
                records.append(
                    {
                        "sequence": sequence,
                        "frame": frame_index,
                        "proposal_id": proposal_id,
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "w": float(box[2] - box[0]),
                        "h": float(box[3] - box[1]),
                    }
                )
                proposal_id += 1
        if position % 100 == 0 or position == len(frame_paths):
            print(f"{sequence}: {position}/{len(frame_paths)} frames", flush=True)
    return pd.DataFrame(records, columns=PROPOSAL_COLUMNS), saturated_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--confirm-frozen-method", action="store_true")
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split == "holdout" and not args.confirm_frozen_method:
        raise ValueError("holdout export requires --confirm-frozen-method")
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    model_config = config["model"]
    settings = config["proposal_export"]
    weights = resolve_path(model_config["weights"])
    if file_sha256(weights) != model_config["expected_sha256"]:
        raise ValueError("model SHA-256 does not match the frozen config")
    dataset_root = resolve_path(dataset["root"])
    sequences_root = dataset_root / dataset["sequences_dir"]
    split_key = f"{args.split}_sequences" if args.split == "development" else "internal_holdout_sequences"
    frozen_sequences = list(dataset[split_key])
    selected = args.sequences or frozen_sequences
    unknown = sorted(set(selected).difference(frozen_sequences))
    if unknown:
        raise ValueError(f"sequences are outside the frozen {args.split} split: {unknown}")
    output = resolve_path(args.output or settings["output_root"])
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    predictions_dir = output / "proposals"
    predictions_dir.mkdir()

    from ultralytics import YOLO
    import torch
    import ultralytics

    model = YOLO(str(weights))
    expected_classes = list(model_config["expected_classes"])
    if list(model.names.values()) != expected_classes:
        raise ValueError(f"unexpected model classes: {model.names}")
    allowed_classes = set(model_config["evaluation_classes"])
    started = time.perf_counter()
    frame_counts: dict[str, int] = {}
    frame_hashes: dict[str, str] = {}
    saturated: dict[str, list[int]] = {}
    proposal_counts: dict[str, int] = {}
    peak_cuda_memory_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for sequence in selected:
        frames = sequence_frames(sequences_root / sequence, args.max_frames)
        frame_counts[sequence] = len(frames)
        frame_hashes[sequence] = file_manifest_sha256(frames, sequences_root)
        proposals, saturated_frames = export_sequence(
            model,
            sequence=sequence,
            frame_paths=frames,
            settings=settings,
            allowed_classes=allowed_classes,
        )
        proposals.to_csv(predictions_dir / f"{sequence}.csv", index=False)
        proposal_counts[sequence] = len(proposals)
        saturated[sequence] = saturated_frames
    if torch.cuda.is_available():
        peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated())
    saturated_count = sum(len(frames) for frames in saturated.values())
    run = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "invalid_max_det_saturation" if saturated_count else "completed",
        "claim_boundary": config["scope"]["claim_boundary"],
        "split": args.split,
        "config": {"path": str(args.config), "sha256": file_sha256(config_path)},
        "model": {"path": model_config["weights"], "sha256": file_sha256(weights)},
        "parameters": {**settings, "max_frames": args.max_frames},
        "dataset": {
            "root": dataset["root"],
            "sequences": selected,
            "frame_counts": frame_counts,
            "frame_manifest_sha256": frame_hashes,
        },
        "result": {
            "elapsed_s": time.perf_counter() - started,
            "proposal_counts": proposal_counts,
            "saturated_frames": saturated,
            "saturated_frame_count": saturated_count,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": git_evidence(),
    }
    temporary = output / "run.json.tmp"
    temporary.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output / "run.json")
    if saturated_count and settings.get("fail_on_saturated_frame", True):
        raise RuntimeError(f"{saturated_count} frames reached max_det; cache is invalid")
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
