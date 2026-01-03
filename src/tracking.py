"""Tracking utilities wrapping Norfair for bounding-box tracking."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
from norfair import Detection as NorfairDetection
from norfair import Tracker

from .detector import Detection


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1) * (yb2 - yb1))
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union if union > 0 else 0.0


def _iou_distance(detection: NorfairDetection, tracked_object) -> float:
    if tracked_object.estimate is None:
        return 1.0
    detected_bbox = detection.points.reshape(-1)
    tracked_bbox = tracked_object.estimate.reshape(-1)
    iou = _bbox_iou(detected_bbox, tracked_bbox)
    return 1 - iou


def build_tracker(distance_threshold: float = 0.7) -> Tracker:
    return Tracker(distance_function=_iou_distance, distance_threshold=distance_threshold)


def detections_to_norfair(detections: Iterable[Detection]) -> List[NorfairDetection]:
    nf_detections: List[NorfairDetection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        nf_detections.append(
            NorfairDetection(points=np.array([[x1, y1], [x2, y2]]), scores=np.array([det.score, det.score]))
        )
    return nf_detections
