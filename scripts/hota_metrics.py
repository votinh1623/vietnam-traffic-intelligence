"""Adapter that feeds this project's GT/prediction DataFrames into TrackEval's
official HOTA implementation, bypassing TrackEval's MOTChallenge file-format
dataset loader (which hardcodes a single 'pedestrian' class and an on-disk
seqmap layout that does not fit this project's multi-class VisDrone setup).

Uses the same per-(sequence, class) split as tracking_metrics.py's
class_accumulators so HOTA is directly comparable to the existing MOTA/IDF1
breakdown, then combines them with TrackEval's own combine_sequences (a
weighted combination, not a plain mean).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from trackeval.metrics import HOTA

BOX_COLUMNS = ["x", "y", "w", "h"]


def validate_iou_threshold(iou_threshold: float) -> None:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the interval (0, 1]")


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Raw (ungated) IoU matrix between xywh boxes; TrackEval applies its own
    alpha-threshold sweep internally, so this must not be pre-thresholded."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))
    gt_x1, gt_y1 = gt_boxes[:, 0], gt_boxes[:, 1]
    gt_x2, gt_y2 = gt_x1 + gt_boxes[:, 2], gt_y1 + gt_boxes[:, 3]
    pr_x1, pr_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
    pr_x2, pr_y2 = pr_x1 + pred_boxes[:, 2], pr_y1 + pred_boxes[:, 3]

    ix1 = np.maximum(gt_x1[:, None], pr_x1[None, :])
    iy1 = np.maximum(gt_y1[:, None], pr_y1[None, :])
    ix2 = np.minimum(gt_x2[:, None], pr_x2[None, :])
    iy2 = np.minimum(gt_y2[:, None], pr_y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    gt_area = np.clip(gt_boxes[:, 2], 0, None) * np.clip(gt_boxes[:, 3], 0, None)
    pr_area = np.clip(pred_boxes[:, 2], 0, None) * np.clip(pred_boxes[:, 3], 0, None)
    union = gt_area[:, None] + pr_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def build_hota_data(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Build the per-sequence `data` dict HOTA.eval_sequence expects."""
    required = {"frame", "id", *BOX_COLUMNS}
    for name, frame in (("ground truth", gt_df), ("predictions", pred_df)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    frames: Iterable[int] = sorted(
        set(gt_df["frame"].astype(int)).union(pred_df["frame"].astype(int))
    )
    gt_unique_ids = sorted(gt_df["id"].astype(int).unique())
    pred_unique_ids = sorted(pred_df["id"].astype(int).unique())
    gt_id_map = {value: index for index, value in enumerate(gt_unique_ids)}
    pred_id_map = {value: index for index, value in enumerate(pred_unique_ids)}

    gt_ids_per_frame: list[np.ndarray] = []
    tracker_ids_per_frame: list[np.ndarray] = []
    similarity_per_frame: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0
    for frame_index in frames:
        gt_frame = gt_df[gt_df["frame"] == frame_index]
        pred_frame = pred_df[pred_df["frame"] == frame_index]
        gt_ids_per_frame.append(
            np.array([gt_id_map[i] for i in gt_frame["id"].astype(int)], dtype=int)
        )
        tracker_ids_per_frame.append(
            np.array([pred_id_map[i] for i in pred_frame["id"].astype(int)], dtype=int)
        )
        similarity_per_frame.append(
            iou_matrix(
                gt_frame[BOX_COLUMNS].to_numpy(dtype=float),
                pred_frame[BOX_COLUMNS].to_numpy(dtype=float),
            )
        )
        num_gt_dets += len(gt_frame)
        num_tracker_dets += len(pred_frame)

    return {
        "num_gt_ids": len(gt_unique_ids),
        "num_tracker_ids": len(pred_unique_ids),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_per_frame,
        "tracker_ids": tracker_ids_per_frame,
        "similarity_scores": similarity_per_frame,
    }


def compute_hota(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """HOTA result dict for one (sequence, class) slice."""
    data = build_hota_data(gt_df, pred_df)
    return HOTA().eval_sequence(data)


def compute_many_hota(results_by_name: dict[str, dict]) -> pd.Series:
    """Combine per-(sequence, class) HOTA results into one OVERALL summary,
    using TrackEval's own weighted combination (not a plain mean)."""
    if not results_by_name:
        raise ValueError("at least one result is required")
    hota = HOTA()
    combined = hota.combine_sequences(results_by_name)
    # HOTA/DetA/AssA etc. are arrays over 19 alpha thresholds (0.05..0.95);
    # report the mean over thresholds, matching TrackEval's own summary field.
    summary = {}
    for field in hota.float_array_fields:
        summary[field] = float(np.mean(combined[field]))
    for field in hota.float_fields:
        summary[field] = float(combined[field])
    return pd.Series(summary)
