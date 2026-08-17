import os
import random
import shutil
from pathlib import Path

# Đường dẫn gốc
image_dir = "datasets/vn_images_temp_v2"
label_dir = "datasets/vn_images_temp_v2_labels"
output_dir = "datasets/vietnam_dataset_v3"  # thư mục dataset mới

# Tạo cấu trúc thư mục
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

# Lấy danh sách tất cả ảnh (không có đuôi)
all_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
random.shuffle(all_images)

# Chia 80% train, 20% val
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]


# Hàm copy ảnh và label tương ứng
def copy_files(image_list, split):
    for img_name in image_list:
        # Copy ảnh
        src_img = os.path.join(image_dir, img_name)
        dst_img = os.path.join(output_dir, split, "images", img_name)
        shutil.copy2(src_img, dst_img)

        # Copy label nếu tồn tại
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(label_dir, label_name)
        dst_label = os.path.join(output_dir, split, "labels", label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            # Nếu không có label thì tạo file rỗng (YOLO coi như không có object)
            open(dst_label, "w").close()


copy_files(train_images, "train")
copy_files(val_images, "val")

# Tạo file data.yaml
yaml_content = f"""
names:
- bus
- car
- motorcycle
- pedestrian
- truck
nc: 5
train: {os.path.join(output_dir, 'train', 'images')}
val: {os.path.join(output_dir, 'val', 'images')}
"""
with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    f.write(yaml_content.strip())

print(f"Dataset mới đã được tạo tại {output_dir}")
print(f"Số ảnh train: {len(train_images)}, val: {len(val_images)}")
