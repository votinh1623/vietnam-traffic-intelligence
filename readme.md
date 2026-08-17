# Tiến trình dự án: Hệ thống giám sát giao thông thông minh (Việt Nam)

## Mục tiêu tổng quát
Xây dựng pipeline AI kết hợp Computer Vision (YOLO, tracking) và LLM để phát hiện, theo dõi, phân tích và cảnh báo tình trạng giao thông từ video UAV/camera tĩnh, tối ưu cho đặc thù giao thông Việt Nam.

---

## 1. Giai đoạn Chuẩn bị & Baseline

### Môi trường
- **Hệ điều hành**: Windows 11, Python 3.10, Conda env `traffic`.
- **Phần cứng**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM).
- **Thư viện chính**: `ultralytics`, `opencv-python`, `pandas`, `numpy`, `motmetrics`.

### Dữ liệu
- **VisDrone2019-DET**: Tải thủ công, tổ chức lại cấu trúc cho YOLO.
- **VisDrone2019-MOT-val**: Sử dụng để đánh giá tracking.

### Huấn luyện Baseline
- **Model**: YOLOv8s (11.1M tham số).
- **Tập huấn luyện**: VisDrone2019-DET (6471 ảnh, 10 class).
- **Chiến lược**: 
  - Pretrained COCO → Fine‑tune trên VisDrone.
  - `imgsz=640`, `batch=8`, `epochs=74` (EarlyStopping với `patience=20`).
- **Kết quả**: 
  - mAP@0.5: **0.389** (tất cả 10 class).
  - Lớp `car`: 0.79, `motor`: 0.451.
- **Mục đích**: Cung cấp backbone mạnh, quen góc nhìn UAV cho bước fine‑tune Việt Nam.

---

## 2. Giai đoạn Fine‑tune cho Dữ liệu Việt Nam

### Thu thập & Gán nhãn
- **Video nguồn**: Tải từ YouTube (giao thông Hà Nội, TP.HCM, UAV).
- **Xử lý**: Cắt frame mỗi **0.5 giây** (`extract_frames.py`). Lần đầu thu được ~1300 ảnh.
- **Gán nhãn**: Sử dụng Roboflow (phiên bản miễn phí).
  - **5 class** tinh gọn: `bus`, `car`, `motorcycle`, `pedestrian`, `truck`.
  - **Quy trình**: Upload toàn bộ ảnh lên Roboflow, sử dụng tính năng **AI labeling** (Label Assist) để tự động gán nhãn. Sau đó rà soát, chỉnh sửa thủ công các box bị sai hoặc thiếu.
  - Tập validation giữ lại 275 ảnh được kiểm tra kỹ để đảm bảo chất lượng đánh giá.
  - Tổng dataset sau khi xuất: ~1300 ảnh (bao gồm train và val).
- **Định dạng xuất**: YOLOv8 (Roboflow Export).

### Fine‑tune Model
- **Pretrained**: `best.pt` từ baseline VisDrone.
- **Tập huấn luyện**: Vietnam dataset (`vietnam_dataset_v2`, 5 class).
- **Kỹ thuật**:
  - **Kiến trúc**: Tự động cắt head 10 class → gắn head 5 class mới.
  - **Transfer Learning**: 
    - `freeze=0` (cho phép toàn bộ model cập nhật).
    - `lr=0.0005` (learning rate thấp để giữ kiến thức cũ).
    - `imgsz=1280` (tăng kích thước ảnh để phát hiện vật thể nhỏ).
    - `augment=mosaic+mixup` (tăng cường dữ liệu mạnh).
  - **Phát hiện chính**: Tăng `imgsz` lên 1280 là yếu tố then chốt giúp model phát hiện xe máy nhỏ tốt hơn hẳn so với 640.
- **Kết quả đánh giá** (trên tập val 275 ảnh):
  - mAP@0.5: **0.745**
  - mAP@0.5:0.95: **0.481**
  - `car`: 0.802, `motorcycle`: 0.731, `bus`: 0.805.
  - Precision: 0.741, Recall: 0.699.

---

## 3. Module hóa & Công cụ

### Tổ chức thư mục

1 anh:
python detect.py "C:\Workspace - Copy\cv\test_image.jpg" --conf 0.5

thu muc anh:
python detect.py "C:\Workspace - Copy\cv\datasets\vn_images_temp_v2" --conf 0.4

video:
python detect.py "C:\Workspace - Copy\cv\datasets\raw_videos\traffic_jam.mp4" --conf 0.3

---

## Research restructuring (2026-08)

The repository is being upgraded into a leakage-controlled, reproducible
deployment-readiness study. Historical metrics and files are preserved, but
the current Vietnam v2 split is marked `legacy_invalid_leakage` and must not be
used for scientific claims.

Current protocols:

- [Dataset integrity](docs/dataset_protocol.md)
- [Multi-model CV + LLM/VLM architecture](docs/multimodel_architecture.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [Audited environment](docs/environment.md)

The historical root command remains supported through a compatibility shim:

```powershell
python detect.py INPUT --conf 0.4
```

No edge-device deployment is claimed. RTX measurements are reported as
edge-oriented optimization and deployment-readiness evidence until a real
target device is benchmarked.

### Leakage-controlled Vietnam v4

The non-destructive v4 materialization contains 819 train, 111 calibration,
108 validation, and 176 test images. A post-build audit found no source-group
or same-frame overlap across splits. All emitted training labels use normalized
YOLO bounding boxes; the legacy polygon labels are converted during build.

The v4 test split is locked by a content-addressed image/label manifest. It is
not used for model, threshold, tracker, prompt, or quantization selection.

V5 supersedes v4 for training after the smoke run exposed 53 exact duplicate
boxes that Ultralytics otherwise removed silently at load time. The source
split is unchanged; v5 records deduplication explicitly and has a new test
lock.
