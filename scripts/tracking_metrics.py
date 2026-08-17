"""Shared, tested MOTChallenge metric construction."""

from __future__ import annotations

from collections.abc import Iterable

import motmetrics as mm
import numpy as np
import pandas as pd


BOX_COLUMNS = ["x", "y", "w", "h"]


def validate_iou_threshold(iou_threshold: float) -> None:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the interval (0, 1]")


def iou_distance_matrix(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return MOT IoU distances, gating pairs below the IoU threshold."""
    validate_iou_threshold(iou_threshold)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))
    # motmetrics already returns 1 - IoU. Its max_iou argument is a maximum
    # distance, so an IoU threshold t maps to max_iou = 1 - t.
    return mm.distances.iou_matrix(
        gt_boxes,
        pred_boxes,
        max_iou=1.0 - iou_threshold,
    )


def build_accumulator(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> mm.MOTAccumulator:
    required = {"frame", "id", *BOX_COLUMNS}
    for name, frame in (("ground truth", gt_df), ("predictions", pred_df)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    validate_iou_threshold(iou_threshold)
    accumulator = mm.MOTAccumulator(auto_id=True)
    frames: Iterable[int] = sorted(
        set(gt_df["frame"].astype(int)).union(pred_df["frame"].astype(int))
    )
    for frame_index in frames:
        gt_frame = gt_df[gt_df["frame"] == frame_index]
        pred_frame = pred_df[pred_df["frame"] == frame_index]
        distances = iou_distance_matrix(
            gt_frame[BOX_COLUMNS].to_numpy(dtype=float),
            pred_frame[BOX_COLUMNS].to_numpy(dtype=float),
            iou_threshold,
        )
        accumulator.update(
            gt_frame["id"].to_numpy(dtype=int),
            pred_frame["id"].to_numpy(dtype=int),
            distances,
        )
    return accumulator


def compute_motchallenge_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    accumulator = build_accumulator(gt_df, pred_df, iou_threshold)
    handler = mm.metrics.create()
    return handler.compute(
        accumulator,
        metrics=mm.metrics.motchallenge_metrics,
        name="acc",
        return_dataframe=True,
    )


def compute_many_motchallenge_metrics(
    accumulators: list[mm.MOTAccumulator], names: list[str]
) -> pd.DataFrame:
    if not accumulators:
        raise ValueError("at least one accumulator is required")
    if len(accumulators) != len(names):
        raise ValueError("accumulators and names must have the same length")
    handler = mm.metrics.create()
    return handler.compute_many(
        accumulators,
        names=names,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
