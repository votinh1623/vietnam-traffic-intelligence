import argparse
import os
import sys
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

def process_image(model, image_path, conf, imgsz, device):
    results = model(image_path, imgsz=imgsz, conf=conf, device=device, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                detections.append({
                    'image_path': image_path,
                    'class': cls_name,
                    'confidence': conf_val,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
    return detections

def process_video(model, video_path, conf, imgsz, device, output_video=None, csv_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
        if writer:
            writer.write(results[0].plot())

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                all_detections.append({
                    'frame': frame_idx,
                    'class': cls_name,
                    'confidence': conf_val,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
        frame_idx += 1
        print(f"Frame {frame_idx}/{total_frames} processed", end='\r')
    cap.release()
    if writer:
        writer.release()
    print()
    return all_detections

def main():
    parser = argparse.ArgumentParser(description="YOLO detection on image(s) or video")
    parser.add_argument('input', help="Path to image file, directory of images, or video file")
    parser.add_argument('--conf', type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument('--model', default="runs/detect/finetune/vietnam_v2/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument('--imgsz', type=int, default=1280, help="Inference image size")
    parser.add_argument('--device', type=int, default=0, help="CUDA device (0 for GPU, -1 for CPU)")
    parser.add_argument('--output-csv', default="detections.csv", help="Output CSV file path")
    parser.add_argument('--output-video', help="Output annotated video file (only for video input)")
    args = parser.parse_args()

    # Kiểm tra model tồn tại
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)

    model = YOLO(args.model)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Phân loại input
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            print(f"Processing video: {input_path}")
            dets = process_video(model, str(input_path), args.conf, args.imgsz, args.device,
                                 output_video=args.output_video, csv_path=args.output_csv)
        else:
            # Ảnh đơn
            print(f"Processing image: {input_path}")
            dets = process_image(model, str(input_path), args.conf, args.imgsz, args.device)
    elif input_path.is_dir():
        # Thư mục ảnh
        print(f"Processing images in directory: {input_path}")
        dets = []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
        for f in input_path.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                print(f"  - {f.name}")
                dets.extend(process_image(model, str(f), args.conf, args.imgsz, args.device))
    else:
        print("Input must be a file or directory.")
        sys.exit(1)

    # Ghi CSV
    if dets:
        df = pd.DataFrame(dets)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(dets)} detections to {args.output_csv}")
    else:
        print("No detections found.")

if __name__ == '__main__':
    main()