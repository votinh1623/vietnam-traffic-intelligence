from ultralytics import YOLO


def main():
    # Load model YOLOv8s pretrained
    model = YOLO("yolo26n.pt")

    # Train
    model.train(
        data="VisDrone.yaml",  # File cấu hình bạn đã tạo
        epochs=100,  # Số epoch tối đa
        patience=20,  # Dừng nếu 20 epoch không cải thiện mAP
        imgsz=640,
        batch=8,  # Giữ 8 để an toàn với VRAM 6GB
        device=0,  # GPU
        workers=0,  # Tránh lỗi multiprocessing trên Windows
        project="baseline",
        name="yolo26n_visdrone",
        exist_ok=True,  # Cho phép ghi đè nếu đã có thư mục
        pretrained=True,
        optimizer="auto",
        lr0=0.01,  # Learning rate mặc định cho YOLOv8
        cos_lr=False,  # Dùng scheduler mặc định
        plots=True,  # Tự động vẽ biểu đồ
        save=True,
        val=True,  # Đánh giá sau mỗi epoch
    )


if __name__ == "__main__":
    main()
