"""Evaluate frame-level and line-crossing vehicle counts from MOT trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.benchmark_tracking import load_visdrone_ground_truth  # noqa: E402
from vn_traffic.analytics import TrafficAnalytics  # noqa: E402
from vn_traffic.config import AnalyticsConfig  # noqa: E402
from vn_traffic.schemas import TrackObservation  # noqa: E402


STANDARD_COLUMNS = ["frame", "id", "class_name", "confidence", "x", "y", "w", "h"]


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
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\0{path.stat().st_size}\0{file_sha256(path)}\n".encode("utf-8")
        )
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


def standardized_ground_truth(path: Path, class_map: dict[int, str]) -> pd.DataFrame:
    frame = load_visdrone_ground_truth(path, class_map).copy()
    frame["confidence"] = 1.0
    return frame[STANDARD_COLUMNS]


def standardized_predictions(path: Path, classes: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(STANDARD_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"prediction file lacks columns {sorted(missing)}: {path}")
    frame = frame[frame["class_name"].isin(classes)].copy()
    return frame[STANDARD_COLUMNS]


def frame_count_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    frame_indices: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    gt_counts = ground_truth.groupby(["frame", "class_name"]).size()
    pred_counts = predictions.groupby(["frame", "class_name"]).size()
    total_errors: list[int] = []
    class_absolute_error = {name: 0 for name in class_names}
    class_gt_total = {name: 0 for name in class_names}
    gt_total = 0
    pred_total = 0
    for frame_index in frame_indices:
        frame_gt = 0
        frame_pred = 0
        for class_name in class_names:
            gt = int(gt_counts.get((frame_index, class_name), 0))
            pred = int(pred_counts.get((frame_index, class_name), 0))
            frame_gt += gt
            frame_pred += pred
            class_absolute_error[class_name] += abs(pred - gt)
            class_gt_total[class_name] += gt
        gt_total += frame_gt
        pred_total += frame_pred
        total_errors.append(frame_pred - frame_gt)
    absolute = [abs(value) for value in total_errors]
    return {
        "frame_count": len(frame_indices),
        "ground_truth_box_total": gt_total,
        "predicted_track_box_total": pred_total,
        "signed_error_total": pred_total - gt_total,
        "mae_vehicles_per_frame": sum(absolute) / len(absolute),
        "rmse_vehicles_per_frame": math.sqrt(
            sum(value * value for value in total_errors) / len(total_errors)
        ),
        "wape": sum(absolute) / gt_total if gt_total else None,
        "per_class_wape": {
            name: (
                class_absolute_error[name] / class_gt_total[name]
                if class_gt_total[name]
                else None
            )
            for name in class_names
        },
    }


def crossing_counts(
    tracks: pd.DataFrame,
    *,
    frame_indices: list[int],
    frame_width: int,
    frame_height: int,
    class_to_id: dict[str, int],
    line_y: float,
    line_tolerance_px: float,
    occupancy_grid_size_px: int,
) -> dict[str, dict[str, int]]:
    analytics = TrafficAnalytics(
        AnalyticsConfig(
            roi_polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            counting_line=((0.0, line_y), (1.0, line_y)),
            line_tolerance_px=line_tolerance_px,
            occupancy_grid_size_px=occupancy_grid_size_px,
        )
    )
    grouped = {int(key): value for key, value in tracks.groupby("frame")}
    for frame_index in frame_indices:
        rows = grouped.get(frame_index)
        observations = []
        if rows is not None:
            for row in rows.itertuples(index=False):
                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        timestamp_s=frame_index / 30.0,
                        track_id=int(row.id),
                        class_id=class_to_id[str(row.class_name)],
                        class_name=str(row.class_name),
                        confidence=float(row.confidence),
                        x1=float(row.x),
                        y1=float(row.y),
                        x2=float(row.x + row.w),
                        y2=float(row.y + row.h),
                    )
                )
        analytics.process(
            frame_index=frame_index,
            timestamp_s=frame_index / 30.0,
            tracks=tuple(observations),
            frame_width=frame_width,
            frame_height=frame_height,
        )
    return analytics.summary()["cumulative_crossings"]


def crossing_error_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_total = sum(int(row["ground_truth_count"]) for row in rows)
    pred_total = sum(int(row["predicted_count"]) for row in rows)
    absolute_error = sum(
        abs(int(row["predicted_count"]) - int(row["ground_truth_count"]))
        for row in rows
    )
    return {
        "cell_count": len(rows),
        "ground_truth_crossings": gt_total,
        "predicted_crossings": pred_total,
        "signed_error": pred_total - gt_total,
        "absolute_error": absolute_error,
        "mae_per_sequence_line_direction_class": absolute_error / len(rows),
        "wape": absolute_error / gt_total if gt_total else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    dataset_root = resolve_path(dataset["root"])
    sequences_root = dataset_root / dataset["sequences_dir"]
    annotations_root = dataset_root / dataset["annotations_dir"]
    class_map = {int(key): str(value) for key, value in dataset["class_map"].items()}
    class_names = list(class_map.values())
    class_to_id = {value: key for key, value in class_map.items()}
    selected_sequences = args.sequences or list(dataset["sequences"])
    unknown = sorted(set(selected_sequences).difference(dataset["sequences"]))
    if unknown:
        raise ValueError(f"unknown or excluded counting sequences: {unknown}")
    git = git_snapshot()
    if (
        args.sequences is None
        and config["evaluation"].get("require_clean_worktree", False)
        and git["dirty"]
    ):
        raise ValueError("full counting benchmark requires a clean worktree")
    output = resolve_path(args.output or config["evaluation"]["output_root"])
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    sequence_data: dict[str, dict[str, Any]] = {}
    annotation_paths = []
    for sequence in selected_sequences:
        frames = sorted(
            (sequences_root / sequence).glob("*.jpg"), key=lambda path: int(path.stem)
        )
        if not frames:
            raise ValueError(f"no frames for sequence {sequence}")
        image = cv2.imread(str(frames[0]))
        if image is None:
            raise ValueError(f"cannot decode {frames[0]}")
        annotation_path = annotations_root / f"{sequence}.txt"
        annotation_paths.append(annotation_path)
        sequence_data[sequence] = {
            "frames": [int(path.stem) for path in frames],
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "ground_truth": standardized_ground_truth(annotation_path, class_map),
        }

    candidate_results = {}
    detail_rows = []
    for candidate_name, candidate in config["candidates"].items():
        run_record = resolve_path(candidate["run_record"])
        if file_sha256(run_record) != candidate["expected_run_sha256"]:
            raise ValueError(f"tracking run hash mismatch for {candidate_name}")
        predictions_dir = resolve_path(candidate["predictions_dir"])
        frame_metrics = []
        candidate_crossing_rows = []
        prediction_paths = []
        for sequence, data in sequence_data.items():
            prediction_path = predictions_dir / f"{sequence}.csv"
            prediction_paths.append(prediction_path)
            predictions = standardized_predictions(prediction_path, set(class_names))
            ground_truth = data["ground_truth"]
            metrics = frame_count_metrics(
                ground_truth,
                predictions,
                frame_indices=data["frames"],
                class_names=class_names,
            )
            metrics["sequence"] = sequence
            frame_metrics.append(metrics)
            for line_y in config["counting"]["normalized_horizontal_lines"]:
                gt_crossings = crossing_counts(
                    ground_truth,
                    frame_indices=data["frames"],
                    frame_width=data["width"],
                    frame_height=data["height"],
                    class_to_id=class_to_id,
                    line_y=float(line_y),
                    line_tolerance_px=float(config["counting"]["line_tolerance_px"]),
                    occupancy_grid_size_px=int(config["counting"]["occupancy_grid_size_px"]),
                )
                pred_crossings = crossing_counts(
                    predictions,
                    frame_indices=data["frames"],
                    frame_width=data["width"],
                    frame_height=data["height"],
                    class_to_id=class_to_id,
                    line_y=float(line_y),
                    line_tolerance_px=float(config["counting"]["line_tolerance_px"]),
                    occupancy_grid_size_px=int(config["counting"]["occupancy_grid_size_px"]),
                )
                for direction in ("up", "down"):
                    for class_name in class_names:
                        row = {
                            "candidate": candidate_name,
                            "sequence": sequence,
                            "line_y": float(line_y),
                            "direction": direction,
                            "class_name": class_name,
                            "ground_truth_count": int(
                                gt_crossings.get(direction, {}).get(class_name, 0)
                            ),
                            "predicted_count": int(
                                pred_crossings.get(direction, {}).get(class_name, 0)
                            ),
                        }
                        candidate_crossing_rows.append(row)
                        detail_rows.append(row)
            print(f"{candidate_name}: {sequence} completed", flush=True)
        total_frames = sum(item["frame_count"] for item in frame_metrics)
        gt_boxes = sum(item["ground_truth_box_total"] for item in frame_metrics)
        absolute_frame_error = sum(
            item["wape"] * item["ground_truth_box_total"] for item in frame_metrics
        )
        candidate_results[candidate_name] = {
            "frame_counting": {
                "sequence_count": len(frame_metrics),
                "frame_count": total_frames,
                "ground_truth_box_total": gt_boxes,
                "macro_sequence_mae_vehicles_per_frame": sum(
                    item["mae_vehicles_per_frame"] for item in frame_metrics
                )
                / len(frame_metrics),
                "micro_wape": absolute_frame_error / gt_boxes,
                "per_sequence": frame_metrics,
            },
            "line_crossing": crossing_error_metrics(candidate_crossing_rows),
            "prediction_manifest_sha256": manifest_sha256(
                prediction_paths, predictions_dir
            ),
            "tracking_run_sha256": file_sha256(run_record),
        }

    detail_path = output / "crossing_counts.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    run = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "status": "completed" if args.sequences is None else "smoke_completed",
        "claim_boundary": config["evaluation"]["claim_boundary"],
        "config": {"path": str(args.config), "sha256": file_sha256(config_path)},
        "dataset": {
            "root": dataset["root"],
            "sequences": selected_sequences,
            "excluded_sequences": dataset["excluded_sequences"],
            "annotation_manifest_sha256": manifest_sha256(
                annotation_paths, annotations_root
            ),
        },
        "counting": config["counting"],
        "candidates": candidate_results,
        "artifacts": {
            "crossing_counts": "crossing_counts.csv",
            "crossing_counts_sha256": file_sha256(detail_path),
        },
        "git": git,
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(candidate_results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
