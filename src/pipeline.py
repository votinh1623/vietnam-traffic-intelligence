import cv2
import numpy as np
from ultralytics import YOLO

# Thay đổi tên file video tại đây để test 2 trường hợp
VIDEO_PATH = "data/traffic_normal.mp4"  # Đổi thành video đường vắng để test YOLO rõ nhất

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Lỗi: Không thể mở video tại đường dẫn {VIDEO_PATH}")
    exit()

# Khởi tạo thuật toán trừ nền MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)
prev_frame_gray = None

# KHỞI TẠO MODEL YOLOV8 (Node D)
# Lần đầu chạy, code này sẽ tự động tải file yolov8n.pt về máy (khoảng 6MB)
model = YOLO("yolov8n.pt")


def check_routing_logic(frame):
    global prev_frame_gray

    # 1. TÍNH MẬT ĐỘ VÀ TỐC ĐỘ (Giữ nguyên như cũ)
    fgmask = fgbg.apply(frame)
    density = np.sum(fgmask == 255) / (frame.shape[0] * frame.shape[1])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame_gray is not None:
        diff = cv2.absdiff(gray, prev_frame_gray)
        motion = np.sum(diff > 25) / (frame.shape[0] * frame.shape[1])
    else:
        motion = 0
    prev_frame_gray = gray

    # 2. LOGIC RẼ NHÁNH (Node C)
    if density > 0.4 and motion < 0.02:
        status = "JAM"
        color = (0, 0, 255)  # Đỏ
    elif density > 0.4 and motion >= 0.02:
        status = "STOP & GO"
        color = (0, 165, 255)  # Cam
    else:
        status = "NORMAL"
        color = (0, 255, 0)  # Xanh

    # 3. XỬ LÝ YOLO (Chỉ chạy khi đường đang di chuyển bình thường)
    if status == "NORMAL":
        # Chạy YOLOv8, chỉ lấy kết quả, không in log ra terminal (verbose=False)
        results = model(frame, verbose=False)

        # Lấy kết quả vẽ bounding box có sẵn của YOLO
        annotated_frame = results[0].plot()

        # Copy lại để vẽ text lên trên
        frame = annotated_frame

    # Vẽ text trạng thái lên màn hình (luôn luôn vẽ để biết hệ thống đang nghĩ gì)
    cv2.putText(
        frame,
        f"Density: {int(density*100)}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Motion: {int(motion*100)}%",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


# Lấy FPS của video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
frame_interval = int(fps / 3)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    # Gọi hàm xử lý
    processed_frame = check_routing_logic(frame)

    cv2.imshow("Traffic Camera - Node D (YOLO)", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
