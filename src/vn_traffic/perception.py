"""Single-model YOLO detection and ByteTrack adapter."""

from __future__ import annotations

from typing import Any

from .config import PipelineConfig
from .schemas import PerceptionResult, TrackObservation


class UltralyticsPerception:
    """Own exactly one YOLO instance for both detection and tracking."""

    def __init__(self, config: PipelineConfig):
        if not config.model.is_file():
            raise FileNotFoundError(f"model file not found: {config.model}")
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(str(config.model))

    def process(
        self, frame: Any, *, frame_index: int, timestamp_s: float
    ) -> PerceptionResult:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            imgsz=self.config.imgsz,
            conf=self.config.confidence,
            iou=self.config.iou,
            max_det=self.config.max_det,
            device=self.config.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        observations: list[TrackObservation] = []
        if boxes is not None and len(boxes):
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(coordinates)
            )
            for track_id, class_id, confidence, box in zip(
                track_ids, class_ids, confidences, coordinates
            ):
                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        track_id=track_id,
                        class_id=class_id,
                        class_name=str(self.model.names[class_id]),
                        confidence=float(confidence),
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    )
                )
        return PerceptionResult(
            annotated_frame=result.plot(
                labels=self.config.show_labels,
                conf=self.config.show_confidence,
                line_width=self.config.line_width,
            ),
            tracks=tuple(observations),
        )
