from ultralytics import YOLO


def main():
    # Đường dẫn tới last.pt của quá trình fine-tune HIỆN TẠI
    checkpoint_path = "runs/detect/finetune/vietnam_v2/weights/last.pt"

    model = YOLO(checkpoint_path)
    model.train(resume=True)


if __name__ == "__main__":
    main()
