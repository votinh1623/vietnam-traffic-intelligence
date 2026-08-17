import numpy as np
from ultralytics import YOLO


class Tracker:
    """
    Bộ theo dõi đối tượng sử dụng ByteTrack tích hợp trong Ultralytics.
    """

    def __init__(self, model_path, imgsz=1280, conf=0.4, device=0):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def track(self, frame, persist=True):
        """
        Chạy tracking trên một frame. Trả về kết quả tracking.
        """
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            persist=persist,
            verbose=False,
        )
        return results

    def extract_tracks(self, results):
        """
        Trích xuất thông tin tracking từ kết quả YOLO.
        Trả về danh sách các dict: {
            'track_id': int,
            'bbox': [x1, y1, x2, y2],
            'class': str,
            'confidence': float
        }
        """
        tracks = []
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.int().tolist()
            confs = results[0].boxes.conf.float().tolist()
            for tid, box, cls, conf in zip(track_ids, boxes, classes, confs):
                tracks.append(
                    {
                        "track_id": tid,
                        "bbox": [int(x) for x in box],
                        "class": self.model.names[cls],
                        "confidence": conf,
                    }
                )
        return tracks

    def plot_tracks(self, frame, results):
        """
        Vẽ bounding box và ID lên frame.
        """
        return results[0].plot()
