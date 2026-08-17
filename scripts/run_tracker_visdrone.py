import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def track_sequence(
    model,
    seq_dir,
    output_file,
    conf=0.4,
    imgsz=1280,
    device=0,
    tracker="bytetrack.yaml",
):
    """Chạy tracker trên một sequence và ghi kết quả ra file."""
    images = sorted([f for f in os.listdir(seq_dir) if f.endswith((".jpg", ".png"))])
    if not images:
        print(f"Không tìm thấy ảnh trong {seq_dir}")
        return

    with open(output_file, "w") as f_out:
        for frame_idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(seq_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            # Tracking với tracker được chỉ định
            results = model.track(
                frame,
                imgsz=imgsz,
                conf=conf,
                device=device,
                persist=True,
                tracker=tracker,
                verbose=False,
            )
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().tolist()
                boxes_xyxy = results[0].boxes.xyxy.tolist()
                confs = results[0].boxes.conf.tolist()
                for tid, box, c in zip(track_ids, boxes_xyxy, confs):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    f_out.write(
                        f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{c:.4f},-1,-1,-1\n"
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/vietnam_finetuned.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--data_root", default="datasets/Visdrone/VisDrone2019-MOT-val/sequences"
    )
    parser.add_argument("--output_dir", default="output_track_mot")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker: bytetrack.yaml, botsort.yaml, or path to custom yaml",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    seq_dirs = [d for d in Path(args.data_root).iterdir() if d.is_dir()]
    Path(args.output_dir).mkdir(exist_ok=True)

    for seq in seq_dirs:
        seq_name = seq.name
        out_file = Path(args.output_dir) / f"{seq_name}.txt"
        print(f"Processing sequence {seq_name}...")
        track_sequence(
            model,
            str(seq),
            str(out_file),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            tracker=args.tracker,
        )
    print("Đã hoàn tất tất cả sequences.")


if __name__ == "__main__":
    main()
