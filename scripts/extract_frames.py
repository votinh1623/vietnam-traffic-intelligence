import cv2
import os
from pathlib import Path

VIDEO_DIR = "datasets/raw_videos"
OUTPUT_DIR = "datasets/vn_images_temp_v2"
FRAME_INTERVAL_SEC = 0.5  # lấy 1 ảnh mỗi 0.5 giây

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_extensions = (".mp4", ".avi", ".mov", ".mkv")
video_files = [
    f for f in Path(VIDEO_DIR).iterdir() if f.suffix.lower() in video_extensions
]

if not video_files:
    print(f"Không tìm thấy video trong {VIDEO_DIR}")
    exit()

total_saved = 0
for video_path in video_files:
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Không mở được video: {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * FRAME_INTERVAL_SEC))

    count = 0
    saved = 0
    print(f"\nĐang xử lý video: {video_path.name} (FPS={fps:.1f})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_name = f"{video_name}_frame_{saved:05d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, frame)
            saved += 1
            total_saved += 1
        count += 1

    cap.release()
    print(f"  -> Đã lưu {saved} ảnh.")

print(f"\nHoàn tất! Tổng {total_saved} ảnh đã lưu trong '{OUTPUT_DIR}'.")
