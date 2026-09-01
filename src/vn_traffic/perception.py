"""Single-model YOLO detection and ByteTrack adapter."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .config import PipelineConfig
from .schemas import PerceptionResult, TrackObservation
from .sizing import adaptive_imgsz, adaptive_line_width


class UltralyticsPerception:
    """Own exactly one YOLO instance for both detection and tracking."""

    def __init__(self, config: PipelineConfig):
        if not config.model.is_file():
            raise FileNotFoundError(f"model file not found: {config.model}")
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(str(config.model))
        # Per-track class-vote history: a track's single-frame class label
        # can flicker (e.g. a truck briefly detected as "car" for one noisy
        # frame) even though the box/motion stays continuous under the same
        # track ID. Reporting and drawing the running-majority class instead
        # of the raw per-frame class removes that flicker without touching
        # the tracker's own identity/motion logic.
        self._track_class_votes: dict[int, Counter[int]] = defaultdict(Counter)

    def process(
        self, frame: Any, *, frame_index: int, timestamp_s: float
    ) -> PerceptionResult:
        height, width = frame.shape[0], frame.shape[1]
        imgsz = self.config.imgsz
        if imgsz is None:
            imgsz = adaptive_imgsz(height, width)
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            imgsz=imgsz,
            conf=self.config.confidence,
            iou=self.config.iou,
            max_det=self.config.max_det,
            agnostic_nms=self.config.agnostic_nms,
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
            smoothed_class_ids: list[int] = []
            for index, (track_id, class_id, confidence, box) in enumerate(
                zip(track_ids, class_ids, confidences, coordinates)
            ):
                if track_id is None:
                    smoothed_class_id = class_id
                else:
                    votes = self._track_class_votes[track_id]
                    votes[class_id] += 1
                    smoothed_class_id = votes.most_common(1)[0][0]
                smoothed_class_ids.append(smoothed_class_id)
                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        track_id=track_id,
                        class_id=smoothed_class_id,
                        class_name=str(self.model.names[smoothed_class_id]),
                        confidence=float(confidence),
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    )
                )
            # boxes.data comes straight out of torch.inference_mode(), whose
            # tensors reject in-place writes -- clone once to a regular
            # tensor so boxes.cls (a view onto boxes.data) becomes writable.
            # Writing the smoothed class here makes result.plot() below draw
            # the smoothed label too, not the raw flickering one.
            boxes.data = boxes.data.clone()
            for index, smoothed_class_id in enumerate(smoothed_class_ids):
                boxes.cls[index] = smoothed_class_id
        line_width = self.config.line_width
        if line_width is None:
            line_width = adaptive_line_width(height, width)
        return PerceptionResult(
            annotated_frame=result.plot(
                labels=self.config.show_labels,
                conf=self.config.show_confidence,
                line_width=line_width,
            ),
            tracks=tuple(observations),
        )
