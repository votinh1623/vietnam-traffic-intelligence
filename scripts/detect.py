import argparse
import os
import sys
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO


def get_next_run_dir(base_dir="output"):
    """Tạo thư mục run tiếp theo: output/run1, output/run2, ..."""
    base = Path(base_dir)
    base.mkdir(exist_ok=True)
    existing = [
        d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run")
    ]
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name[3:]))
        except ValueError:
            pass
    next_num = max(numbers) + 1 if numbers else 1
    run_dir = base / f"run{next_num}"
    run_dir.mkdir(parents=True)
    return run_dir


def process_image(model, image_path, conf, imgsz, device, output_dir):
    results = model(image_path, imgsz=imgsz, conf=conf, device=device, verbose=False)
    detections = []
    for r in results:
        # Lưu ảnh đã plot
        out_img_path = output_dir / Path(image_path).name
        r.save(filename=str(out_img_path))
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                detections.append(
                    {
                        "image_path": str(Path(image_path).resolve()),
                        "class": cls_name,
                        "confidence": conf_val,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
    return detections


def process_video(model, video_path, conf, imgsz, device, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = output_dir / Path(video_path).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
        writer.write(results[0].plot())
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                all_detections.append(
                    {
                        "frame": frame_idx,
                        "class": cls_name,
                        "confidence": conf_val,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total_frames}", end="\r")
    cap.release()
    writer.release()
    print()
    return all_detections


def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection - outputs results with auto-run folders"
    )
    parser.add_argument("input", help="Path to image, video, or directory of images")
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--model",
        default="runs/detect/finetune/vietnam_v2/weights/best.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device (0 for GPU, -1 for CPU)"
    )
    args = parser.parse_args()

    # Kiểm tra model
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)

    model = YOLO(args.model)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Tạo thư mục output/runN
    run_dir = get_next_run_dir("output")
    print(f"Saving results to: {run_dir}")

    detections = []

    if input_path.is_file():
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"Processing video: {input_path}")
            detections = process_video(
                model, input_path, args.conf, args.imgsz, args.device, run_dir
            )
        else:
            print(f"Processing image: {input_path}")
            detections = process_image(
                model, input_path, args.conf, args.imgsz, args.device, run_dir
            )
    elif input_path.is_dir():
        print(f"Processing images in directory: {input_path}")
        # Lưu ảnh vào run_dir/images (để tránh lẫn lộn)
        img_out_dir = run_dir / "images"
        img_out_dir.mkdir(exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        for f in input_path.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                print(f"  - {f.name}")
                dets = process_image(
                    model, str(f), args.conf, args.imgsz, args.device, img_out_dir
                )
                detections.extend(dets)
    else:
        print("Input must be a file or directory.")
        sys.exit(1)

    # Ghi CSV vào run_dir
    if detections:
        csv_path = run_dir / "detections.csv"
        df = pd.DataFrame(detections)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(detections)} detections to {csv_path}")
    else:
        print("No detections found.")


if __name__ == "__main__":
    main()
