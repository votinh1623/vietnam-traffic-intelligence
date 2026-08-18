"""Run a provenance-recorded class-aware tracking benchmark on VisDrone MOT."""

from __future__ import annotations

import argparse
import hashlib
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

from tracking_metrics import build_accumulator, compute_many_motchallenge_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_COLUMNS = [
    "frame",
    "id",
    "x",
    "y",
    "w",
    "h",
    "score",
    "class_id",
    "truncation",
    "occlusion",
]
TRACK_COLUMNS = [
    "sequence",
    "frame",
    "id",
    "class_name",
    "confidence",
    "x",
    "y",
    "w",
    "h",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest_sha256(paths: list[Path], root: Path) -> str:
    """Hash ordered file identities, sizes, and content hashes."""
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        record = f"{relative}\0{path.stat().st_size}\0{file_sha256(path)}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_visdrone_ground_truth(
    path: Path, class_map: dict[int, str]
) -> pd.DataFrame:
    """Load valid target rows from the native ten-column VisDrone MOT format."""
    frame = pd.read_csv(path, header=None, names=GT_COLUMNS)
    frame = frame[
        (frame["score"] == 1)
        & frame["class_id"].isin(class_map)
        & (frame["w"] > 0)
        & (frame["h"] > 0)
    ].copy()
    frame["class_name"] = frame["class_id"].map(class_map)
    return frame[["frame", "id", "class_name", "x", "y", "w", "h"]]


def sequence_frames(sequence_dir: Path, max_frames: int | None = None) -> list[Path]:
    frames = sorted(
        sequence_dir.glob("*.jpg"), key=lambda path: int(path.stem)
    )
    if not frames:
        raise ValueError(f"sequence contains no JPG frames: {sequence_dir}")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    return frames


def reset_ultralytics_tracker(model: Any) -> None:
    """Reset persistent tracker state before starting an independent sequence."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()) or ():
        tracker.reset()


def track_sequence(
    model: Any,
    *,
    sequence: str,
    frame_paths: list[Path],
    perception: dict[str, Any],
) -> pd.DataFrame:
    reset_ultralytics_tracker(model)
    records: list[dict[str, Any]] = []
    for position, frame_path in enumerate(frame_paths, start=1):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"cannot decode frame: {frame_path}")
        result = model.track(
            image,
            persist=True,
            tracker=str(resolve_path(perception["tracker"])),
            imgsz=int(perception["imgsz"]),
            conf=float(perception["confidence"]),
            iou=float(perception["iou"]),
            device=str(perception["device"]),
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            for track_id, class_id, confidence, box in zip(
                ids, classes, confidences, coordinates
            ):
                records.append(
                    {
                        "sequence": sequence,
                        "frame": int(frame_path.stem),
                        "id": track_id,
                        "class_name": str(model.names[class_id]),
                        "confidence": float(confidence),
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "w": float(box[2] - box[0]),
                        "h": float(box[3] - box[1]),
                    }
                )
        if position % 100 == 0 or position == len(frame_paths):
            print(f"{sequence}: {position}/{len(frame_paths)} frames", flush=True)
    return pd.DataFrame(records, columns=TRACK_COLUMNS)


def class_accumulators(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    sequence: str,
    class_names: list[str],
    iou_threshold: float,
) -> tuple[list[Any], list[str]]:
    accumulators = []
    names = []
    for class_name in class_names:
        gt_class = ground_truth[ground_truth["class_name"] == class_name]
        pred_class = predictions[predictions["class_name"] == class_name]
        accumulators.append(
            build_accumulator(gt_class, pred_class, iou_threshold=iou_threshold)
        )
        names.append(f"{sequence}:{class_name}")
    return accumulators, names


def git_evidence() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *args],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

    status = run("status", "--porcelain").splitlines()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def json_metric_values(series: pd.Series) -> dict[str, int | float | None]:
    """Convert pandas/numpy metric scalars into strict JSON values."""
    values: dict[str, int | float | None] = {}
    for name, value in series.items():
        if pd.isna(value):
            values[name] = None
        elif hasattr(value, "item"):
            values[name] = value.item()
        else:
            values[name] = value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    model_config = config["model"]
    evaluation = config["evaluation"]
    dataset_root = resolve_path(dataset["root"])
    sequences_root = dataset_root / dataset["sequences_dir"]
    annotations_root = dataset_root / dataset["annotations_dir"]
    weights = resolve_path(model_config["weights"])
    if file_sha256(weights) != model_config["expected_sha256"]:
        raise ValueError("model SHA-256 does not match the benchmark config")
    class_map = {int(key): value for key, value in dataset["class_map"].items()}
    class_names = list(model_config["expected_classes"])

    available = sorted(path.name for path in sequences_root.iterdir() if path.is_dir())
    selected = args.sequences or (
        available if evaluation["sequences"] == "all" else evaluation["sequences"]
    )
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"unknown sequences: {unknown}")
    output = resolve_path(args.output or evaluation["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    import torch
    import ultralytics

    model = YOLO(str(weights))
    if list(model.names.values()) != class_names:
        raise ValueError(f"unexpected model classes: {model.names}")
    all_accumulators: list[Any] = []
    accumulator_names: list[str] = []
    annotation_hashes: dict[str, str] = {}
    frame_manifest_hashes: dict[str, str] = {}
    sequence_frame_counts: dict[str, int] = {}
    started = time.perf_counter()
    for sequence in selected:
        annotation_path = annotations_root / f"{sequence}.txt"
        annotation_hashes[sequence] = file_sha256(annotation_path)
        frame_paths = sequence_frames(sequences_root / sequence, args.max_frames)
        sequence_frame_counts[sequence] = len(frame_paths)
        frame_manifest_hashes[sequence] = file_manifest_sha256(
            frame_paths, sequences_root
        )
        predictions = track_sequence(
            model,
            sequence=sequence,
            frame_paths=frame_paths,
            perception=config["perception"],
        )
        predictions.to_csv(predictions_dir / f"{sequence}.csv", index=False)
        ground_truth = load_visdrone_ground_truth(annotation_path, class_map)
        evaluated_frames = {int(path.stem) for path in frame_paths}
        ground_truth = ground_truth[ground_truth["frame"].isin(evaluated_frames)]
        accumulators, names = class_accumulators(
            ground_truth,
            predictions,
            sequence=sequence,
            class_names=class_names,
            iou_threshold=float(evaluation["metric_iou_threshold"]),
        )
        all_accumulators.extend(accumulators)
        accumulator_names.extend(names)

    summary = compute_many_motchallenge_metrics(all_accumulators, accumulator_names)
    summary.to_csv(output / "metrics.csv")
    elapsed_s = time.perf_counter() - started
    run = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "status": "completed",
        "claim_boundary": evaluation["claim_boundary"],
        "config": {"path": str(args.config), "sha256": file_sha256(config_path)},
        "model": {"path": model_config["weights"], "sha256": file_sha256(weights)},
        "dataset": {
            "root": dataset["root"],
            "annotation_sha256": annotation_hashes,
            "frame_manifest_sha256": frame_manifest_hashes,
            "sequence_frame_counts": sequence_frame_counts,
            "class_map": class_map,
        },
        "parameters": {
            **config["perception"],
            "tracker_sha256": file_sha256(
                resolve_path(config["perception"]["tracker"])
            ),
            "metric_iou_threshold": evaluation["metric_iou_threshold"],
            "max_frames": args.max_frames,
        },
        "result": {
            "elapsed_s": elapsed_s,
            "metrics_path": "metrics.csv",
            "overall": json_metric_values(summary.loc["OVERALL"]),
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
    print(summary.loc[["OVERALL"]].to_string())
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
