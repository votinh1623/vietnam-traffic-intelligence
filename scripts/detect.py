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


def _adaptive_line_width(height, width):
    # line_width is an absolute pixel value passed straight to Ultralytics'
    # Annotator (font scale = line_width/3, no independent font-size
    # control) -- a fixed value tuned to look right on ~720-1280px video
    # renders as illegibly thin/small on much higher-resolution input (e.g.
    # 4K), since box/vehicle pixel size grows with resolution too. Scale it
    # off the frame's short side so it stays proportionally consistent
    # across resolutions; 720p is the regime it was originally tuned on.
    return max(1, round(min(height, width) / 720))


def _adaptive_imgsz(height, width):
    # A fixed imgsz=1280 letterboxes a 3840x2160 source down ~3x before the
    # detector ever sees it, shrinking already-small overhead vehicles below
    # what the model can reliably detect (measured: 14 boxes at imgsz=1280
    # vs 53 at imgsz=2560 on the same 4K frame, for ~0.4GB extra VRAM --
    # well inside a 6GB budget). Scale to the source's long side, clamped to
    # [1280, 2560]: unchanged for the ~720-1280px videos this was
    # originally tuned on, larger only for sources that actually need it.
    long_side = max(height, width)
    clamped = max(1280, min(2560, long_side))
    return -(-clamped // 32) * 32


def process_image(model, image_path, conf, imgsz, device, output_dir):
    if imgsz is None:
        probe = cv2.imread(str(image_path))
        if probe is None:
            raise ValueError(f"Cannot read image: {image_path}")
        imgsz = _adaptive_imgsz(probe.shape[0], probe.shape[1])
    # agnostic_nms=True: without it, NMS only suppresses overlapping boxes
    # within the same predicted class, so one real vehicle can end up boxed
    # twice under two different class labels (e.g. "truck 0.85" + "car
    # 0.58" on the same object) instead of keeping only the higher-
    # confidence box.
    results = model(
        image_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        agnostic_nms=True,
        verbose=False,
    )
    detections = []
    for r in results:
        # Lưu ảnh đã plot -- line_width scales with resolution so id/class/
        # conf labels stay legible without covering small boxes on any
        # input size (same convention as the main pipeline's perception.py).
        out_img_path = output_dir / Path(image_path).name
        line_width = _adaptive_line_width(*r.orig_shape)
        r.save(filename=str(out_img_path), line_width=line_width)
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
    line_width = _adaptive_line_width(h, w)
    if imgsz is None:
        imgsz = _adaptive_imgsz(h, w)

    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(
            frame,
            imgsz=imgsz,
            conf=conf,
            device=device,
            agnostic_nms=True,
            verbose=False,
        )
        writer.write(results[0].plot(line_width=line_width))
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
        default="runs/detect/research/yolov8s_v6_seed0-2/weights/best.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default: auto, adaptive to source resolution)",
    )
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
