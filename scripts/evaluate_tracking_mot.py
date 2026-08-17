import argparse
import pandas as pd
import motmetrics as mm
import numpy as np
from pathlib import Path


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
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
            dists = 1.0 - iou
        else:
            dists = np.zeros((len(gt_boxes), len(pred_boxes)))
        acc.update(
            gt_frame["id"].astype(int).values,
            pred_frame["id"].astype(int).values,
            dists,
        )
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=mm.metrics.motchallenge_metrics, return_dataframe=True
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="datasets/Visdrone/VisDrone2019-MOT-val/annotations")
    parser.add_argument("--pred_dir", default="output_track_mot")
    parser.add_argument("--output_metrics", default="tracking_metrics_visdrone.csv")
    parser.add_argument(
        "--classes",
        type=str,
        default="1,4,6,9,10",
        help="Comma-separated VisDrone class IDs (default: 1,4,6,9,10)",
    )
    args = parser.parse_args()

    valid_classes = [int(c) for c in args.classes.split(",")]
    print(f"Đánh giá với các class VisDrone: {valid_classes}")

    gt_files = sorted(Path(args.gt_dir).glob("*.txt"))
    all_summaries = []
    for gt_file in gt_files:
        seq_name = gt_file.stem
        pred_file = Path(args.pred_dir) / f"{seq_name}.txt"
        if not pred_file.exists():
            print(f"Không tìm thấy dự đoán cho {seq_name}, bỏ qua.")
            continue
        print(f"Đánh giá {seq_name}...")
        gt = load_gt(str(gt_file), valid_classes=valid_classes)
        pred = load_pred(str(pred_file))
        summary = compute_metrics(gt, pred)
        summary["sequence"] = seq_name
        all_summaries.append(summary)

    if all_summaries:
        total_summary = pd.concat(all_summaries)
        # Tính trung bình
        avg = total_summary.mean(numeric_only=True).to_frame().T
        avg["sequence"] = "Average"
        total_summary = pd.concat([total_summary, avg], ignore_index=True)
        total_summary.to_csv(args.output_metrics, index=False)
        print("\n======= Tracking Evaluation trên VisDrone-MOT =======")
        print(total_summary[["sequence", "mota", "motp", "idf1", "num_switches"]])
    else:
        print("Không có dữ liệu để đánh giá.")


if __name__ == "__main__":
    main()
