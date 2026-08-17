import os
from pathlib import Path
from PIL import Image

# Đường dẫn gốc
base = Path("datasets/VisDrone")
train_img = base / "VisDrone2019-DET-train/images"
train_ann = base / "VisDrone2019-DET-train/annotations"
val_img = base / "VisDrone2019-DET-val/images"
val_ann = base / "VisDrone2019-DET-val/annotations"


def convert_visdrone_to_yolo(ann_dir, img_dir, out_dir):
    out_dir.mkdir(exist_ok=True)
    for ann_file in ann_dir.glob("*.txt"):
        img_file = img_dir / (ann_file.stem + ".jpg")
        if not img_file.exists():
            continue  # bỏ qua nếu không tìm thấy ảnh

        # Đọc kích thước ảnh thật
        with Image.open(str(img_file)) as img:
            img_w, img_h = img.size

        with open(ann_file, "r") as f:
            lines = f.readlines()

        yolo_lines = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            # VisDrone format: bbox_left,bbox_top,bbox_width,bbox_height,score,class,trunc,occlusion
            x1 = int(parts[0])
            y1 = int(parts[1])
            w = int(parts[2])
            h = int(parts[3])
            cls = int(parts[5])

            # Bỏ qua class 0 (ignored regions) và các class không hợp lệ
            if cls == 0 or cls > 10:
                continue
            cls = cls - 1  # về 0-9 # VisDrone đánh class từ 1, YOLO từ 0

            # Tính tâm và chuẩn hóa
            x_center = (x1 + w / 2) / img_w
            y_center = (y1 + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            yolo_lines.append(
                f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        # Ghi file label
        out_file = out_dir / ann_file.name
        with open(out_file, "w") as f:
            f.write("\n".join(yolo_lines))


# Chạy convert
print("Đang chuyển đổi tập train...")
convert_visdrone_to_yolo(train_ann, train_img, base / "VisDrone2019-DET-train/labels")
print("Đang chuyển đổi tập val...")
convert_visdrone_to_yolo(val_ann, val_img, base / "VisDrone2019-DET-val/labels")
print("Hoàn tất! Đã tạo labels.")
