import argparse
import pandas as pd
from pathlib import Path

from tracking_metrics import (
    build_accumulator,
    compute_many_motchallenge_metrics,
    compute_motchallenge_metrics,
)


def load_gt(
    gt_file, valid_classes=[1, 4, 6, 9, 10]
):  # pedestrian, car, truck, bus, motor
    """Đọc GT file VisDrone-MOT, trả về DataFrame với cột frame, id, x, y, w, h"""
    df = pd.read_csv(
        gt_file,
        header=None,
        names=["frame", "id", "x", "y", "w", "h", "score", "class", "trunc", "occl"],
    )
    # Lọc chỉ giữ score=1 (object thực) và class nằm trong valid_classes
    df = df[(df["score"] == 1) & (df["class"].isin(valid_classes))]
    return df[["frame", "id", "x", "y", "w", "h"]]


def load_pred(pred_file):
    """Đọc file dự đoán MOT, trả về DataFrame với cột frame, id, x, y, w, h"""
    df = pd.read_csv(
        pred_file,
        header=None,
        names=["frame", "id", "x", "y", "w", "h", "conf", "z1", "z2", "z3"],
    )
    return df[["frame", "id", "x", "y", "w", "h"]]


def compute_metrics(gt_df, pred_df, iou_threshold=0.5):
    return compute_motchallenge_metrics(gt_df, pred_df, iou_threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir", default="datasets/Visdrone/VisDrone2019-MOT-val/annotations"
    )
    parser.add_argument("--pred_dir", default="output_track_mot")
    parser.add_argument("--output_metrics", default="tracking_metrics_visdrone.csv")
    parser.add_argument("--iou", type=float, default=0.5, help="Minimum match IoU")
    parser.add_argument(
        "--classes",
        type=str,
        default="1,4,6,9,10",
        help="Comma-separated VisDrone class IDs (default: 1,4,6,9,10)",
    )
    args = parser.parse_args()

    valid_classes = [int(c) for c in args.classes.split(",")]
    print(f"Evaluating VisDrone classes: {valid_classes}")

    gt_files = sorted(Path(args.gt_dir).glob("*.txt"))
    accumulators = []
    sequence_names = []
    for gt_file in gt_files:
        seq_name = gt_file.stem
        pred_file = Path(args.pred_dir) / f"{seq_name}.txt"
        if not pred_file.exists():
            print(f"Prediction not found for {seq_name}; skipping.")
            continue
        print(f"Evaluating {seq_name}...")
        gt = load_gt(str(gt_file), valid_classes=valid_classes)
        pred = load_pred(str(pred_file))
        accumulators.append(build_accumulator(gt, pred, args.iou))
        sequence_names.append(seq_name)

    if accumulators:
        total_summary = compute_many_motchallenge_metrics(
            accumulators, sequence_names
        )
        total_summary.index.name = "sequence"
        total_summary.to_csv(args.output_metrics)
        print("\n======= VisDrone-MOT Tracking Evaluation =======")
        print(total_summary[["mota", "motp", "idf1", "num_switches"]])
    else:
        print("No matching ground-truth and prediction files were found.")


if __name__ == "__main__":
    main()
