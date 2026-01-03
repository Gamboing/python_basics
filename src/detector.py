"""YOLOv8 ONNX-based detector using onnxruntime for CPU inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    """Single object detection result."""

    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


class YoloV8OnnxDetector:
    """Minimal YOLOv8 ONNX runtime wrapper for CPU-only inference."""

    def __init__(
        self,
        model_path: Path,
        class_names: Sequence[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = list(class_names)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, frame: np.ndarray, img_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and normalize frame for YOLOv8 ONNX model."""
        height, width = frame.shape[:2]
        scale = img_size / max(height, width)
        new_width, new_height = int(width * scale), int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        canvas[: new_height, : new_width] = resized
        image = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        return image, 1 / scale, (new_width, new_height)

    def __call__(self, frame: np.ndarray, img_size: int) -> List[Detection]:
        blob, gain, _ = self._preprocess(frame, img_size)
        ort_inputs = {self.input_name: blob}
        preds = self.session.run(None, ort_inputs)[0]
        preds = np.squeeze(preds, axis=0)
        boxes = preds[:, :4]
        scores = preds[:, 4:5] * preds[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences > self.conf_threshold
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
        if boxes.size == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes)
        nms_indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        detections: List[Detection] = []
        for idx in nms_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx] / gain
            class_id = int(class_ids[idx])
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(confidences[idx]),
                    class_id=class_id,
                    class_name=self.class_names[class_id] if class_id < len(self.class_names) else str(class_id),
                )
            )
        return detections

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        return np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Simple non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]

        return keep


# Local import to avoid circular dependency in type checkers
import cv2  # noqa: E402  # isort:skip
