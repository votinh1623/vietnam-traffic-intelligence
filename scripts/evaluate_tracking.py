import argparse
import pandas as pd

from tracking_metrics import compute_motchallenge_metrics


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
    """Compute CLEAR MOT and identity metrics using IoU-distance matching."""
    return compute_motchallenge_metrics(gt_df, pred_df, iou_threshold)


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
    print("\nMetric directions:")
    print("- MOTA and IDF1: higher is better")
    print("- MOTP: mean matched 1-IoU distance; lower is better")
    print("- num_switches: lower is better")


if __name__ == "__main__":
    main()
