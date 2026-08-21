"""Measure the incremental oracle value of proposals missed by ByteTrack."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from benchmark_tracking import GT_COLUMNS, file_sha256, resolve_path


GT_DETAIL_COLUMNS = [
    "sequence", "frame", "id", "class_name", "x", "y", "w", "h",
    "truncation", "occlusion",
]


def iou_matrix(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    if left.empty or right.empty:
        return np.zeros((len(left), len(right)), dtype=float)
    a = left[["x", "y", "w", "h"]].to_numpy(dtype=float)
    b = right[["x", "y", "w", "h"]].to_numpy(dtype=float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    inter_w = np.maximum(0.0, np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0, None], b[:, 0]))
    inter_h = np.maximum(0.0, np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1, None], b[:, 1]))
    intersection = inter_w * inter_h
    union = a[:, 2, None] * a[:, 3, None] + b[:, 2] * b[:, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def maximum_cardinality_matches(matrix: np.ndarray, threshold: float) -> dict[int, int]:
    """Return deterministic maximum-cardinality GT-to-prediction matches."""
    adjacency = [
        [int(j) for j in np.argsort(-row) if row[j] >= threshold]
        for row in matrix
    ]
    prediction_to_gt: dict[int, int] = {}

    def augment(gt_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[gt_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = prediction_to_gt.get(prediction_index)
            if previous is None or augment(previous, visited):
                prediction_to_gt[prediction_index] = gt_index
                return True
        return False

    for gt_index in range(matrix.shape[0]):
        augment(gt_index, set())
    return {gt: prediction for prediction, gt in prediction_to_gt.items()}


def load_detailed_gt(path: Path, sequence: str, class_map: dict[int, str]) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=GT_COLUMNS)
    frame = frame[
        (frame["score"] == 1)
        & frame["class_id"].isin(class_map)
        & (frame["w"] > 0)
        & (frame["h"] > 0)
    ].copy()
    frame["class_name"] = frame["class_id"].map(class_map)
    frame["sequence"] = sequence
    return frame[GT_DETAIL_COLUMNS]


def classify_observations(
    ground_truth: pd.DataFrame,
    baseline: pd.DataFrame,
    proposals: pd.DataFrame,
    *,
    iou_threshold: float,
    track_low_thresh: float,
    track_high_thresh: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["sequence", "frame", "class_name"]
    baseline_groups = {key: value.reset_index(drop=True) for key, value in baseline.groupby(group_columns)}
    proposal_groups = {key: value.reset_index(drop=True) for key, value in proposals.groupby(group_columns)}
    for key, gt_group in ground_truth.groupby(group_columns, sort=True):
        gt_group = gt_group.reset_index(drop=True)
        baseline_group = baseline_groups.get(key, pd.DataFrame(columns=["x", "y", "w", "h"]))
        baseline_matches = maximum_cardinality_matches(
            iou_matrix(gt_group, baseline_group), iou_threshold
        )
        missed_indices = [index for index in range(len(gt_group)) if index not in baseline_matches]
        proposal_group = proposal_groups.get(key, pd.DataFrame(columns=["x", "y", "w", "h", "confidence"]))
        missed_gt = gt_group.iloc[missed_indices].reset_index(drop=True)
        proposal_matches = maximum_cardinality_matches(
            iou_matrix(missed_gt, proposal_group), iou_threshold
        )
        missed_to_proposal = {
            missed_indices[missed_index]: proposal_index
            for missed_index, proposal_index in proposal_matches.items()
        }
        for gt_index, gt in gt_group.iterrows():
            caught = gt_index in baseline_matches
            proposal_index = missed_to_proposal.get(gt_index)
            candidate = proposal_index is not None
            confidence = float(proposal_group.iloc[proposal_index]["confidence"]) if candidate else None
            if caught:
                status = "bytetrack_caught"
                score_band = None
            elif not candidate:
                status = "no_post_nms_proposal"
                score_band = None
            elif confidence < track_low_thresh:
                status = "incremental_candidate"
                score_band = "below_track_low"
            elif confidence < track_high_thresh:
                status = "incremental_candidate"
                score_band = "second_stage_range"
            else:
                status = "incremental_candidate"
                score_band = "at_or_above_track_high"
            row = gt.to_dict()
            row.update(
                {
                    "status": status,
                    "proposal_confidence": confidence,
                    "proposal_score_band": score_band,
                    "sqrt_area_px": math.sqrt(float(gt.w) * float(gt.h)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_temporal_support(observations: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    result = observations.copy()
    evidence = result[result["status"] != "no_post_nms_proposal"]
    frame_lookup = {
        (str(sequence), int(track_id)): set(group["frame"].astype(int))
        for (sequence, track_id), group in evidence.groupby(["sequence", "id"])
    }
    for window in windows:
        before_values = []
        after_values = []
        for row in result.itertuples(index=False):
            frames = frame_lookup.get((str(row.sequence), int(row.id)), set())
            frame = int(row.frame)
            before_values.append(any(frame - distance in frames for distance in range(1, window + 1)))
            after_values.append(any(frame + distance in frames for distance in range(1, window + 1)))
        result[f"support_before_{window}"] = before_values
        result[f"support_after_{window}"] = after_values
        result[f"support_one_sided_{window}"] = np.logical_or(before_values, after_values)
        result[f"support_forward_backward_{window}"] = np.logical_and(before_values, after_values)
    return result


def wape_from_counts(gt_counts: pd.Series, pred_counts: pd.Series) -> float:
    indices = gt_counts.index.union(pred_counts.index)
    gt = gt_counts.reindex(indices, fill_value=0)
    pred = pred_counts.reindex(indices, fill_value=0)
    return float((pred - gt).abs().sum() / gt.sum())


def summarize(
    observations: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    windows: list[int],
    tiny_max_sqrt_area_px: float,
    occluded_min_level: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    # Tracking exports contain every VisDrone class, while this study is scoped
    # to vehicles. Keep count metrics on exactly the GT class universe.
    vehicle_classes = set(observations["class_name"].astype(str))
    baseline = baseline[baseline["class_name"].isin(vehicle_classes)].copy()
    total = len(observations)
    caught = observations["status"] == "bytetrack_caught"
    incremental = observations["status"] == "incremental_candidate"
    difficult_miss = (~caught) & (
        (observations["sqrt_area_px"] < tiny_max_sqrt_area_px)
        | (observations["occlusion"] >= occluded_min_level)
    )
    difficult_recoverable = difficult_miss & incremental
    gt_counts = observations.groupby(["sequence", "frame"]).size()
    pred_counts = baseline.groupby(["sequence", "frame"]).size()
    recovered_counts = observations[incremental].groupby(["sequence", "frame"]).size()
    oracle_counts = pred_counts.add(recovered_counts, fill_value=0)
    baseline_wape = wape_from_counts(gt_counts, pred_counts)
    oracle_wape = wape_from_counts(gt_counts, oracle_counts)
    recovery_fraction = float(difficult_recoverable.sum() / difficult_miss.sum()) if difficult_miss.sum() else 0.0
    recall_gain = float(incremental.sum() / total) if total else 0.0
    wape_reduction = float((baseline_wape - oracle_wape) / baseline_wape) if baseline_wape else 0.0
    temporal = {}
    temporal_by_score_band: dict[str, dict[str, Any]] = {}
    for window in windows:
        target = observations[incremental]
        temporal[str(window)] = {
            "one_sided_count": int(target[f"support_one_sided_{window}"].sum()),
            "forward_backward_count": int(target[f"support_forward_backward_{window}"].sum()),
        }
    score_bands = (
        observations[incremental]["proposal_score_band"].value_counts().sort_index().astype(int).to_dict()
    )
    for score_band in score_bands:
        band = observations[incremental & (observations["proposal_score_band"] == score_band)]
        temporal_by_score_band[score_band] = {
            str(window): {
                "candidate_count": len(band),
                "one_sided_count": int(band[f"support_one_sided_{window}"].sum()),
                "forward_backward_count": int(band[f"support_forward_backward_{window}"].sum()),
            }
            for window in windows
        }
    per_sequence: dict[str, dict[str, Any]] = {}
    for sequence, sequence_observations in observations.groupby("sequence"):
        sequence_baseline = baseline[baseline["sequence"] == sequence]
        sequence_caught = sequence_observations["status"] == "bytetrack_caught"
        sequence_incremental = sequence_observations["status"] == "incremental_candidate"
        sequence_difficult = (~sequence_caught) & (
            (sequence_observations["sqrt_area_px"] < tiny_max_sqrt_area_px)
            | (sequence_observations["occlusion"] >= occluded_min_level)
        )
        sequence_gt_counts = sequence_observations.groupby(["sequence", "frame"]).size()
        sequence_pred_counts = sequence_baseline.groupby(["sequence", "frame"]).size()
        sequence_recovered = sequence_observations[sequence_incremental].groupby(["sequence", "frame"]).size()
        sequence_oracle_counts = sequence_pred_counts.add(sequence_recovered, fill_value=0)
        sequence_baseline_wape = wape_from_counts(sequence_gt_counts, sequence_pred_counts)
        sequence_oracle_wape = wape_from_counts(sequence_gt_counts, sequence_oracle_counts)
        per_sequence[str(sequence)] = {
            "ground_truth_observations": len(sequence_observations),
            "bytetrack_detection_recall": float(sequence_caught.sum() / len(sequence_observations)),
            "incremental_candidates": int(sequence_incremental.sum()),
            "oracle_detection_recall_absolute_gain": float(sequence_incremental.sum() / len(sequence_observations)),
            "difficult_incremental_fraction": (
                float((sequence_difficult & sequence_incremental).sum() / sequence_difficult.sum())
                if sequence_difficult.sum()
                else 0.0
            ),
            "baseline_frame_count_wape": sequence_baseline_wape,
            "oracle_frame_count_wape": sequence_oracle_wape,
        }
    gate_results = {
        "incremental_candidate_fraction_tiny_or_occluded": recovery_fraction >= float(gates["incremental_candidate_fraction_tiny_or_occluded_min"]),
        "oracle_detection_recall_or_wape": (
            recall_gain >= float(gates["oracle_detection_recall_absolute_gain_min"])
            or wape_reduction >= float(gates["oracle_frame_count_wape_relative_reduction_min"])
        ),
        "nonzero_temporal_support": (
            any(value["one_sided_count"] > 0 for value in temporal.values())
            if gates.get("require_nonzero_temporal_support", True)
            else True
        ),
    }
    return {
        "ground_truth_observations": total,
        "bytetrack_caught": int(caught.sum()),
        "bytetrack_detection_recall": float(caught.sum() / total) if total else 0.0,
        "incremental_candidates": int(incremental.sum()),
        "no_post_nms_proposal": int((observations["status"] == "no_post_nms_proposal").sum()),
        "incremental_score_bands": score_bands,
        "difficult_bytetrack_misses": int(difficult_miss.sum()),
        "difficult_incremental_candidates": int(difficult_recoverable.sum()),
        "incremental_candidate_fraction_tiny_or_occluded": recovery_fraction,
        "oracle_detection_recall_absolute_gain": recall_gain,
        "baseline_frame_count_wape": baseline_wape,
        "oracle_frame_count_wape": oracle_wape,
        "oracle_frame_count_wape_relative_reduction": wape_reduction,
        "temporal_support": temporal,
        "temporal_support_by_score_band": temporal_by_score_band,
        "per_sequence": per_sequence,
        "gates": gate_results,
        "proceed_to_stage_c": all(gate_results.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    baseline_config = config["bytetrack_baseline"]
    oracle_config = config["oracle"]
    proposal_root = resolve_path(args.proposals or config["proposal_export"]["output_root"])
    proposal_run = json.loads((proposal_root / "run.json").read_text(encoding="utf-8"))
    if proposal_run["status"] != "completed" or proposal_run["split"] != "development":
        raise ValueError("oracle requires a completed development proposal export")
    config_sha256 = file_sha256(config_path)
    if proposal_run["config"]["sha256"] != config_sha256:
        raise ValueError("proposal export does not match the current frozen config")
    if proposal_run["parameters"].get("max_frames") is not None:
        raise ValueError("oracle rejects partial/smoke proposal exports")
    if proposal_run["dataset"]["sequences"] != list(dataset["development_sequences"]):
        raise ValueError("proposal export does not cover the frozen development split")
    baseline_run = resolve_path(baseline_config["run_record"])
    if file_sha256(baseline_run) != baseline_config["expected_run_sha256"]:
        raise ValueError("ByteTrack baseline run hash mismatch")
    root = resolve_path(dataset["root"])
    class_map = {int(key): str(value) for key, value in dataset["class_map"].items()}
    sequences = list(dataset["development_sequences"])
    gt_frames = []
    baseline_frames = []
    proposal_frames = []
    for sequence in sequences:
        gt_frames.append(load_detailed_gt(root / dataset["annotations_dir"] / f"{sequence}.txt", sequence, class_map))
        baseline_frames.append(pd.read_csv(resolve_path(baseline_config["predictions_dir"]) / f"{sequence}.csv"))
        proposal_frames.append(pd.read_csv(proposal_root / "proposals" / f"{sequence}.csv"))
    ground_truth = pd.concat(gt_frames, ignore_index=True)
    baseline = pd.concat(baseline_frames, ignore_index=True)
    proposals = pd.concat(proposal_frames, ignore_index=True)
    observations = classify_observations(
        ground_truth,
        baseline,
        proposals,
        iou_threshold=float(oracle_config["iou_threshold"]),
        track_low_thresh=float(baseline_config["track_low_thresh"]),
        track_high_thresh=float(baseline_config["track_high_thresh"]),
    )
    windows = [int(value) for value in oracle_config["temporal_windows"]]
    observations = add_temporal_support(observations, windows)
    result = summarize(
        observations,
        baseline,
        windows=windows,
        tiny_max_sqrt_area_px=float(oracle_config["tiny_max_sqrt_area_px"]),
        occluded_min_level=int(oracle_config["occluded_min_level"]),
        gates=config["gates"],
    )
    output = resolve_path(args.output or oracle_config["output_root"])
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    observations.to_csv(output / "observations.csv", index=False)
    report = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "completed",
        "claim_boundary": config["scope"]["claim_boundary"],
        "config": {"path": str(args.config), "sha256": config_sha256},
        "proposal_run": {"path": str(proposal_root / "run.json"), "sha256": file_sha256(proposal_root / "run.json")},
        "bytetrack_run": {"path": baseline_config["run_record"], "sha256": file_sha256(baseline_run)},
        "result": result,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
