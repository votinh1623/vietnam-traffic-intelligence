from ultralytics import YOLO
import numpy as np


class Detector:
    def __init__(self, model_path, imgsz=1280, conf=0.4, device=0):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, image):
        """Trả về toàn bộ kết quả inference (có thể dùng để plot hoặc lấy boxes)"""
        return self.model(
            image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )

    def get_boxes(self, results):
        """Trích xuất boxes dạng numpy array [x1,y1,x2,y2,conf,cls]"""
        if not results or len(results) == 0:
            return np.empty((0, 6))
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf, cls])
