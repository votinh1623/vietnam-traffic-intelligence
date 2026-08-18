"""Compare standard and SAHI inference on VisDrone small objects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_COLUMNS = 8


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        record = (
            f"{path.relative_to(root).as_posix()}\0{path.stat().st_size}\0"
            f"{file_sha256(path)}\n"
        )
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def parse_visdrone_annotation(
    path: Path, *, width: int, height: int, class_count: int
) -> list[dict[str, Any]]:
    """Return valid boxes; score-zero/ignored regions are deliberately excluded."""
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.rstrip(",").split(",")]
        if len(parts) != ANNOTATION_COLUMNS:
            raise ValueError(f"{path}:{line_number} has {len(parts)} columns")
        left, top, box_width, box_height = map(float, parts[:4])
        score, category_id = int(parts[4]), int(parts[5])
        if score != 1 or not 1 <= category_id <= class_count:
            continue
        x1 = max(0.0, left)
        y1 = max(0.0, top)
        x2 = min(float(width), left + box_width)
        y2 = min(float(height), top + box_height)
        if x2 <= x1 or y2 <= y1:
            continue
        annotations.append(
            {
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
        )
    return annotations


def build_coco_ground_truth(
    image_paths: list[Path], annotations_dir: Path, class_names: list[str]
) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        height, width = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"annotation unavailable: {annotation_path}")
        for annotation in parse_visdrone_annotation(
            annotation_path,
            width=width,
            height=height,
            class_count=len(class_names),
        ):
            annotation.update({"id": annotation_id, "image_id": image_id})
            annotations.append(annotation)
            annotation_id += 1
    return {
        "info": {"description": "VisDrone valid-object COCO-style evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(class_names, start=1)
        ],
    }


def standard_predictions(
    *,
    image_paths: list[Path],
    image_ids: dict[str, int],
    weights: Path,
    model_config: dict[str, Any],
    mode: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], int]:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    predictions = []
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    for index, image_path in enumerate(image_paths, start=1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(
            str(image_path),
            imgsz=int(mode["imgsz"]),
            conf=float(model_config["confidence"]),
            iou=float(model_config["iou"]),
            max_det=int(model_config["max_det"]),
            device=model_config["device"],
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        boxes = result.boxes
        if boxes is not None:
            for xyxy, score, class_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.int().cpu().tolist(),
            ):
                predictions.append(
                    {
                        "image_id": image_ids[image_path.name],
                        "category_id": class_id + 1,
                        "bbox": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2] - xyxy[0]),
                            float(xyxy[3] - xyxy[1]),
                        ],
                        "score": float(score),
                    }
                )
        if index % 50 == 0 or index == len(image_paths):
            print(f"standard: {index}/{len(image_paths)} images", flush=True)
    return predictions, latencies, int(torch.cuda.max_memory_allocated())


def sliced_predictions(
    *,
    image_paths: list[Path],
    image_ids: dict[str, int],
    weights: Path,
    model_config: dict[str, Any],
    mode: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], int]:
    import torch
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=float(model_config["confidence"]),
        device=model_config["device"],
        image_size=int(mode["image_size"]),
    )
    predictions = []
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    for index, image_path in enumerate(image_paths, start=1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=int(mode["slice_height"]),
            slice_width=int(mode["slice_width"]),
            overlap_height_ratio=float(mode["overlap_height_ratio"]),
            overlap_width_ratio=float(mode["overlap_width_ratio"]),
            perform_standard_pred=bool(mode["perform_standard_pred"]),
            postprocess_type=mode["postprocess_type"],
            postprocess_match_metric=mode["postprocess_match_metric"],
            postprocess_match_threshold=float(mode["postprocess_match_threshold"]),
            postprocess_class_agnostic=False,
            force_postprocess_type=bool(mode["force_postprocess_type"]),
            verbose=0,
            auto_slice_resolution=False,
            batch_size=int(mode["batch_size"]),
        )
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        for prediction in result.object_prediction_list:
            bbox = prediction.bbox.to_xywh()
            predictions.append(
                {
                    "image_id": image_ids[image_path.name],
                    "category_id": int(prediction.category.id) + 1,
                    "bbox": [float(value) for value in bbox],
                    "score": float(prediction.score.value),
                }
            )
        if index % 25 == 0 or index == len(image_paths):
            print(f"sahi: {index}/{len(image_paths)} images", flush=True)
    return predictions, latencies, int(torch.cuda.max_memory_allocated())


def evaluate_coco(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    max_dets: list[int],
) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [image["id"] for image in ground_truth["images"]]
    evaluator.params.catIds = [category["id"] for category in ground_truth["categories"]]
    evaluator.params.maxDets = max_dets
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]

    def mean_valid(values: Any) -> float | None:
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None

    def iou_index(value: float) -> int:
        matches = np.flatnonzero(np.isclose(evaluator.params.iouThrs, value))
        if not matches.size:
            raise ValueError(f"COCO evaluator does not contain IoU threshold {value}")
        return int(matches[0])

    last_max_det = len(max_dets) - 1
    per_class_ap = {}
    per_class_ap_small = {}
    for class_index, category in enumerate(ground_truth["categories"]):
        per_class_ap[category["name"]] = mean_valid(
            precision[:, :, class_index, 0, last_max_det]
        )
        per_class_ap_small[category["name"]] = mean_valid(
            precision[:, :, class_index, 1, last_max_det]
        )
    return {
        "max_dets": int(max_dets[last_max_det]),
        "ap": mean_valid(precision[:, :, :, 0, last_max_det]),
        "ap50": mean_valid(precision[iou_index(0.50), :, :, 0, last_max_det]),
        "ap75": mean_valid(precision[iou_index(0.75), :, :, 0, last_max_det]),
        "ap_small": mean_valid(precision[:, :, :, 1, last_max_det]),
        "ap_medium": mean_valid(precision[:, :, :, 2, last_max_det]),
        "ap_large": mean_valid(precision[:, :, :, 3, last_max_det]),
        "ar_max_dets": mean_valid(recall[:, :, 0, last_max_det]),
        "per_class_ap": per_class_ap,
        "per_class_ap_small": per_class_ap_small,
    }


def validate_config(config: dict[str, Any], mode_name: str) -> None:
    if config.get("dataset", {}).get("split") != "validation":
        raise ValueError("SAHI selection benchmark only accepts split: validation")
    if mode_name not in config.get("modes", {}):
        raise ValueError(f"unknown evaluation mode: {mode_name}")
    mode_type = config["modes"][mode_name].get("type")
    if mode_type not in {"standard", "sahi"}:
        raise ValueError(f"unsupported mode type: {mode_type}")
    if mode_type == "sahi":
        force_postprocess_type = config["modes"][mode_name]["force_postprocess_type"]
        if (
            float(config["model"]["confidence"]) < 0.1
            and config["modes"][mode_name]["postprocess_type"] != "NMS"
            and not force_postprocess_type
        ):
            raise ValueError(
                "low-confidence SAHI evaluation must force its declared postprocess type"
            )


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *args],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

    status = run("status", "--porcelain").splitlines()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "ultralytics", "sahi", "pycocotools", "opencv-python", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, args.mode)
    mode = config["modes"][args.mode]
    images_dir = resolve_path(config["dataset"]["images"])
    annotations_dir = resolve_path(config["dataset"]["annotations"])
    weights = resolve_path(config["model"]["weights"])
    if file_sha256(weights) != config["model"]["expected_sha256"]:
        raise ValueError("model SHA-256 mismatch")
    image_paths = sorted(images_dir.glob("*.jpg"))
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("max_images must be positive")
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError("no VisDrone validation images found")
    git = git_snapshot()
    if (
        args.max_images is None
        and config["evaluation"]["require_clean_worktree"]
        and git["dirty"]
    ):
        raise ValueError("full SAHI benchmark requires a clean worktree")
    output = resolve_path(
        args.output
        or Path(config["evaluation"]["output_root"]) / args.mode
    )
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    ground_truth = build_coco_ground_truth(
        image_paths, annotations_dir, config["dataset"]["classes"]
    )
    ground_truth_path = output / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(ground_truth, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    image_ids = {image["file_name"]: image["id"] for image in ground_truth["images"]}
    if mode["type"] == "standard":
        predictions, latencies, peak_vram = standard_predictions(
            image_paths=image_paths,
            image_ids=image_ids,
            weights=weights,
            model_config=config["model"],
            mode=mode,
        )
    else:
        predictions, latencies, peak_vram = sliced_predictions(
            image_paths=image_paths,
            image_ids=image_ids,
            weights=weights,
            model_config=config["model"],
            mode=mode,
        )
    predictions_path = output / "predictions.json"
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = evaluate_coco(
        ground_truth,
        predictions,
        list(config["evaluation"]["coco_max_dets"]),
    )
    latency = np.asarray(latencies, dtype=float)
    annotation_paths = [annotations_dir / f"{path.stem}.txt" for path in image_paths]
    import torch

    record = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "status": "completed" if args.max_images is None else "smoke_completed",
        "claim_boundary": config["evaluation"]["claim_boundary"],
        "mode": args.mode,
        "mode_config": mode,
        "image_count": len(image_paths),
        "ground_truth_box_count": len(ground_truth["annotations"]),
        "prediction_count": len(predictions),
        "metrics": metrics,
        "latency_s_per_image": {
            "mean": float(latency.mean()),
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
        },
        "peak_vram_allocated_bytes": peak_vram,
        "provenance": {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": file_sha256(config_path),
            "model_sha256": file_sha256(weights),
            "image_manifest_sha256": manifest_sha256(image_paths, images_dir),
            "annotation_manifest_sha256": manifest_sha256(annotation_paths, annotations_dir),
            "ground_truth_sha256": file_sha256(ground_truth_path),
            "predictions_sha256": file_sha256(predictions_path),
            "git": git,
            "packages": package_versions(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
