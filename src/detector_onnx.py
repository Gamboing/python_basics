from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.config import DetectorConfig


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    cls: int


def letterbox(image: "np.ndarray", size: int) -> Tuple["np.ndarray", float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def non_max_suppression(dets: "np.ndarray", iou_thresh: float) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class OnnxDetector:
    def __init__(self, model_path: str, config: DetectorConfig):
        self.config = config
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions(),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image: "np.ndarray") -> Tuple["np.ndarray", float, Tuple[int, int]]:
        img, scale, pad = letterbox(image, self.config.imgsz)
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img, scale, pad

    def postprocess(
        self, output: Sequence["np.ndarray"], scale: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        preds = output[0]
        if preds.ndim == 3:
            preds = preds[0]
        detections: List[Detection] = []

        # Expecting [x1,y1,x2,y2,score,class]
        if preds.shape[1] < 6:
            return detections

        mask = preds[:, 4] >= self.config.conf
        preds = preds[mask]
        if preds.size == 0:
            return detections

        keep = non_max_suppression(preds, self.config.iou)
        for i in keep:
            x1, y1, x2, y2, score, cls = preds[i, :6]
            x1 = (x1 - pad[0]) / scale
            x2 = (x2 - pad[0]) / scale
            y1 = (y1 - pad[1]) / scale
            y2 = (y2 - pad[1]) / scale
            x1 = np.clip(x1, 0, orig_shape[1])
            x2 = np.clip(x2, 0, orig_shape[1])
            y1 = np.clip(y1, 0, orig_shape[0])
            y2 = np.clip(y2, 0, orig_shape[0])
            detections.append(Detection(bbox=(x1, y1, x2, y2), score=float(score), cls=int(cls)))
        return detections

    def __call__(self, image: "np.ndarray") -> List[Detection]:
        input_blob, scale, pad = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_blob})
        return self.postprocess(outputs, scale, pad, image.shape[:2])
