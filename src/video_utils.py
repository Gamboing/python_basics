"""Video utility helpers for resizing, drawing, and logging."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection


@dataclass
class FrameTimings:
    frame_index: int
    detection_ms: float
    tracking_ms: float
    render_ms: float


@dataclass
class RunStats:
    total_frames: int
    processed_frames: int
    avg_fps: float
    avg_detection_ms: float
    avg_tracking_ms: float
    avg_render_ms: float


COCO_CLASSES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_resize(frame: np.ndarray, target_width: Optional[int]) -> np.ndarray:
    if target_width is None:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / width
    new_dim = (target_width, int(height * scale))
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)


def draw_detections(frame: np.ndarray, detections: Iterable[Detection], track_ids: List[int]) -> np.ndarray:
    annotated = frame.copy()
    for det, track_id in zip(detections, track_ids):
        x1, y1, x2, y2 = map(int, det.bbox)
        color = (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.score:.2f} id={track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 6), (x1 + w + 2, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return annotated


def write_tracks_csv(
    csv_path: Path,
    rows: Iterable[Tuple[int, float, int, str, float, float, float, float, float]],
) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_sec", "track_id", "class_name", "conf", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)


def write_run_log(log_path: Path, stats: RunStats) -> None:
    with log_path.open("w") as f:
        f.write("Run summary\n")
        f.write(f"Total frames: {stats.total_frames}\n")
        f.write(f"Processed frames: {stats.processed_frames}\n")
        f.write(f"Average FPS: {stats.avg_fps:.2f}\n")
        f.write(f"Avg detection time (ms): {stats.avg_detection_ms:.2f}\n")
        f.write(f"Avg tracking time (ms): {stats.avg_tracking_ms:.2f}\n")
        f.write(f"Avg render/write time (ms): {stats.avg_render_ms:.2f}\n")


def compute_run_stats(timings: List[FrameTimings], total_frames: int, processed_frames: int, total_seconds: float) -> RunStats:
    if not timings:
        return RunStats(0, 0, 0.0, 0.0, 0.0, 0.0)
    avg_det = sum(t.detection_ms for t in timings) / len(timings)
    avg_track = sum(t.tracking_ms for t in timings) / len(timings)
    avg_render = sum(t.render_ms for t in timings) / len(timings)
    fps = processed_frames / total_seconds if total_seconds > 0 else 0.0
    return RunStats(total_frames, processed_frames, fps, avg_det, avg_track, avg_render)


def timestamp_from_frame(frame_idx: int, fps: float) -> float:
    return frame_idx / fps if fps > 0 else 0.0


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    return wrapper
