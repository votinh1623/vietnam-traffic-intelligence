import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import pandas as pd
from src.detection.tracker import Tracker


def get_next_run_dir(base_dir="output_track"):
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


def main():
    parser = argparse.ArgumentParser(description="Test tracking on video")
    parser.add_argument("input", help="Path to video file")
    parser.add_argument("--model", default="models/vietnam_finetuned.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)
    if not Path(args.input).exists():
        print(f"Video not found: {args.input}")
        sys.exit(1)

    tracker = Tracker(args.model, imgsz=args.imgsz, conf=args.conf, device=args.device)
    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    run_dir = get_next_run_dir("output_track")
    output_video = run_dir / Path(args.input).name
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    all_tracks = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = tracker.track(frame)
        annotated = tracker.plot_tracks(frame, results)
        writer.write(annotated)

        tracks = tracker.extract_tracks(results)
        for t in tracks:
            t["frame"] = frame_idx
        all_tracks.extend(tracks)

        frame_idx += 1
        print(f"Frame {frame_idx}/{total_frames}", end="\r")

    cap.release()
    writer.release()

    if all_tracks:
        csv_path = run_dir / "tracks.csv"
        pd.DataFrame(all_tracks).to_csv(csv_path, index=False)
        print(f"\nSaved {len(all_tracks)} track entries to {csv_path}")
    print(f"Output video saved to {output_video}")


if __name__ == "__main__":
    main()
