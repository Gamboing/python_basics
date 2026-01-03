from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detector_onnx import Detection
from src.config import TrackerConfig


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    cls: int
    hits: int = 1
    misses: int = 0
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)


def bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union


class IoUTracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        if not detections and not self.tracks:
            return []

        det_bboxes = [det.bbox for det in detections]
        det_used = [False] * len(detections)

        # Asociar detecciones a tracks existentes por IoU máximo
        for track in list(self.tracks.values()):
            best_iou = 0.0
            best_j = -1
            for j, bbox in enumerate(det_bboxes):
                if det_used[j]:
                    continue
                iou = bbox_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= self.config.iou_match and best_j >= 0:
                det = detections[best_j]
                track.history.append(track.bbox)
                track.bbox = det.bbox
                track.score = det.score
                track.cls = det.cls
                track.hits += 1
                track.misses = 0
                det_used[best_j] = True
            else:
                track.misses += 1

        # Crear tracks nuevos para detecciones no usadas
        for j, used in enumerate(det_used):
            if not used:
                det = detections[j]
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    score=det.score,
                    cls=det.cls,
                )
                self.next_id += 1

        # Filtrar tracks desaparecidos
        self.tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if t.misses <= self.config.max_missed and t.hits >= self.config.min_hits
        }
        return list(self.tracks.values())
