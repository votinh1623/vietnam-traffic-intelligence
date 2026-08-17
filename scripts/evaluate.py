import argparse
import sys
import os
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá model YOLO trên tập validation/test"
    )
    parser.add_argument(
        "--model", required=True, help="Đường dẫn file .pt đã fine-tune"
    )
    parser.add_argument(
        "--data", required=True, help="Đường dẫn file data.yaml của dataset"
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output", default="evaluation_results", help="Thư mục lưu kết quả đánh giá"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Không tìm thấy model: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Không tìm thấy data.yaml: {args.data}")
        sys.exit(1)

    model = YOLO(args.model)
    # Chạy validation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,  # vẽ confusion matrix, PR curve, F1 curve...
        save_json=False,  # nếu cần json kết quả thì bật True
        project=args.output,
        name="val",
    )

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("=" * 50)
    print(f"Biểu đồ và kết quả chi tiết được lưu tại: {args.output}/val/")


if __name__ == "__main__":
    main()
