from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.config import PoseConfig


@dataclass
class PoseResult:
    keypoints: List[Tuple[float, float, float]]  # (x, y, score)

    def wrists(self) -> List[Tuple[float, float, float]]:
        # COCO indices: 9 (left wrist), 10 (right wrist)
        wrists_idx = [9, 10]
        return [self.keypoints[i] for i in wrists_idx if i < len(self.keypoints)]


class OnnxPoseEstimator:
    def __init__(self, model_path: str, config: PoseConfig):
        self.config = config
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions(),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image: "np.ndarray") -> Tuple["np.ndarray", float, Tuple[int, int]]:
        h, w = image.shape[:2]
        size = self.config.imgsz
        scale = min(size / h, size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img, scale, (left, top)

    def postprocess(self, outputs: Sequence["np.ndarray"], scale: float, pad: Tuple[int, int]) -> PoseResult:
        preds = outputs[0]  # (1, K, 3) expected
        if preds.ndim == 3:
            preds = preds[0]
        keypoints: List[Tuple[float, float, float]] = []
        for x, y, score in preds:
            x = (x - pad[0]) / scale
            y = (y - pad[1]) / scale
            keypoints.append((float(x), float(y), float(score)))
        return PoseResult(keypoints=keypoints)

    def __call__(self, image: "np.ndarray") -> PoseResult:
        blob, scale, pad = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs, scale, pad)
