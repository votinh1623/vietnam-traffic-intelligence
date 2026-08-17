import os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Cấu hình
image_dir = "datasets/vn_images_temp_v2"  # Thư mục chứa 1300 ảnh
output_label_dir = "datasets/vn_images_temp_v2_labels"  # Nơi lưu file label YOLO
os.makedirs(output_label_dir, exist_ok=True)

# Các class của bạn – phải khớp với thứ tự trong data.yaml sau này
class_names = ["bus", "car", "motorcycle", "pedestrian", "truck"]
# Tạo text prompt: các class cách nhau bởi dấu chấm (theo yêu cầu của Grounding DINO)
text_prompt = ". ".join(class_names) + "."

# Thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tải model và processor (chỉ tải một lần)
print("Đang tải Grounding DINO...")
model_id = "IDEA-Research/grounding-dino-tiny"  # phiên bản nhẹ
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Ngưỡng confidence để giữ lại box
confidence_threshold = 0.3


# Hàm chuyển đổi box từ định dạng (center_x, center_y, width, height) tỉ lệ [0,1] sang YOLO
def convert_to_yolo(box, img_width, img_height):
    # box ở đây là [x_center, y_center, width, height] đã chuẩn hóa (theo tỉ lệ 0-1)
    x_center, y_center, w, h = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


# Lấy danh sách ảnh
image_extensions = (".jpg", ".jpeg", ".png")
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
print(f"Tìm thấy {len(image_files)} ảnh.")

# Xử lý từng ảnh
for idx, img_name in enumerate(image_files):
    img_path = os.path.join(image_dir, img_name)
    # Đọc ảnh bằng PIL
    image = Image.open(img_path).convert("RGB")
    img_width, img_height = image.size

    # Tiền xử lý và dự đoán
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Hậu xử lý kết quả
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=confidence_threshold,
        text_threshold=confidence_threshold,
        target_sizes=[(img_height, img_width)],
    )[0]

    # Lưu nhãn vào file txt
    label_path = os.path.join(output_label_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(label_path, "w") as f:
        for box, label in zip(results["boxes"], results["labels"]):
            # box: tensor [x1, y1, x2, y2] dạng pixel (góc trên trái, góc dưới phải)
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            # Chuẩn hóa
            x_center /= img_width
            y_center /= img_height
            w /= img_width
            h /= img_height

            # Lấy class_id từ nhãn text
            label_str = label.lower()
            try:
                class_id = class_names.index(label_str)  # tìm đúng index (0-4)
            except ValueError:
                continue  # bỏ qua nếu không khớp class nào

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    if (idx + 1) % 50 == 0:
        print(f"Đã xử lý {idx+1}/{len(image_files)} ảnh")

print("Hoàn tất! Các file label được lưu tại:", output_label_dir)
