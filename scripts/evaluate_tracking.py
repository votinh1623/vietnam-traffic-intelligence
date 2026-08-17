import argparse
import pandas as pd
import motmetrics as mm
import numpy as np


def load_ground_truth(gt_path):
    """Đọc ground truth CSV và trả về DataFrame với cột: frame, id, x, y, w, h"""
    df = pd.read_csv(gt_path)
    # Giả sử gt có các cột: frame, track_id, x1, y1, x2, y2
    df["x"] = df["x1"]
    df["y"] = df["y1"]
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["id"] = df["track_id"]
    return df[["frame", "id", "x", "y", "w", "h"]]


def load_predictions(pred_path):
    """Tương tự cho file dự đoán"""
    df = pd.read_csv(pred_path)
    df["x"] = df["x1"]
    df["y"] = df["y1"]
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["id"] = df["track_id"]
    return df[["frame", "id", "x", "y", "w", "h"]]


def compute_metrics(gt_df, pred_df, iou_threshold=0.5):
    """Tính MOTA, MOTP, IDF1, HOTA (nếu có)"""
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]
        # Tính khoảng cách IoU giữa các cặp box
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        # Dùng hàm distances: 1 - IoU
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
            dists = 1.0 - iou
        else:
            dists = np.zeros((len(gt_boxes), len(pred_boxes)))
        acc.update(
            gt_frame["id"].values.astype(int),
            pred_frame["id"].values.astype(int),
            dists,
        )
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=mm.metrics.motchallenge_metrics, return_dataframe=True
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking results")
    parser.add_argument("--gt", required=True, help="Ground truth CSV file")
    parser.add_argument("--pred", required=True, help="Predicted tracks CSV file")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    args = parser.parse_args()

    gt = load_ground_truth(args.gt)
    pred = load_predictions(args.pred)
    summary = compute_metrics(gt, pred, args.iou)
    print("\n======= Tracking Evaluation =======")
    print(
        summary[
            [
                "mota",
                "motp",
                "idf1",
                "num_switches",
                "mostly_tracked",
                "partially_tracked",
                "mostly_lost",
            ]
        ]
    )
    print("\nGiải thích:")
    print("- MOTA: Multi-Object Tracking Accuracy (cao = tốt, thường > 50% là khá)")
    print("- MOTP: Precision về định vị (thấp = tốt)")
    print("- IDF1: Tỉ lệ nhận dạng đúng (cao = tốt)")
    print("- num_switches: số lần đổi ID (càng thấp càng tốt)")


if __name__ == "__main__":
    main()
