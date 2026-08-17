import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.preprocessing.density_router import DensityRouter


def main():
    video_path = "C:/Workspace - Copy/cv/datasets/raw_videos/YTDown.com_YouTube_Crazy-rush-hour-traffic-in-Saigon-Ho-Chi_Media_V8paX22Sgzg_001_1080p.mp4"
    output_path = "output/density_router_test_v2.mp4"
    Path("output").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không mở được video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    router = DensityRouter(
        density_threshold=0.85,
        grid_rows=8,
        grid_cols=10,
        cell_density_threshold=0.7,
        cell_ratio_threshold=0.8,
        confirm_time=10.0,
        normal_confirm_time=5.0,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        mode, density, cell_ratio, fg_mask = router.route(frame, timestamp)

        if frame_idx % 50 == 0:
            print(
                f"Frame {frame_idx}: Density={density:.2f}, Cells={cell_ratio:.2f}, Mode={mode}"
            )

        color = (0, 0, 255) if mode == "JAM" else (0, 255, 0)
        cv2.putText(
            frame, f"Mode: {mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
        )
        cv2.putText(
            frame,
            f"Density: {density:.2f}  Cells: {cell_ratio:.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Hiển thị fg_mask ở góc trên bên trái để debug
        if fg_mask is not None:
            mask_small = cv2.resize(fg_mask, (w // 4, h // 4))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[0 : h // 4, 0 : w // 4] = mask_color

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video kết quả đã lưu tại: {output_path}")
    print(f"Tổng frame đã xử lý: {frame_idx}")


if __name__ == "__main__":
    main()
