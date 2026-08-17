from ultralytics import YOLO


def main():
    # Load baseline VisDrone
    model = YOLO("runs/detect/baseline/yolov8s_visdrone/weights/best.pt")

    # Fine‑tune trên dữ liệu Việt Nam
    model.train(
        data="datasets/vietnam_dataset_v2/data.yaml",  # đường dẫn tới file yaml của bạn
        epochs=30,  # tối đa 30 epoch
        patience=15,  # dừng nếu không cải thiện sau 10 epoch
        imgsz=1280,
        batch=8,  # batch nhỏ vì ít ảnh
        device=0,
        workers=0,  # tránh lỗi multiprocessing
        lr0=0.0005,  # learning rate thấp
        freeze=10,  # đóng băng backbone
        augment=True,
        mosaic=1.0,
        mixup=0.3,  # tăng cường mixup
        project="finetune",
        name="vietnam_v2",
        exist_ok=True,
        pretrained=True,
        plots=True,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()
