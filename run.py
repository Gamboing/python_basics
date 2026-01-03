from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from src.detector import YoloV8OnnxDetector
from src.tracking import build_tracker, detections_to_norfair
from src.video_utils import (
    COCO_CLASSES,
    FrameTimings,
    compute_run_stats,
    draw_detections,
    ensure_dir,
    maybe_resize,
    timestamp_from_frame,
    write_run_log,
    write_tracks_csv,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only YOLOv8 ONNX detection + Norfair tracking")
    parser.add_argument("--video-path", required=True, help="Path to local input video")
    parser.add_argument("--model-path", default="models/yolov8n.onnx", help="Path to YOLOv8n ONNX model")
    parser.add_argument("--output-dir", default="outputs", help="Directory for annotated video and logs")
    parser.add_argument("--resize", type=int, default=None, help="Optional target width to downscale frames (e.g., 1280)")
    parser.add_argument("--every-n-frames", type=int, default=1, help="Process every Nth frame to speed up (e.g., 2 or 3)")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size for YOLOv8 (square)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--distance-threshold", type=float, default=0.7, help="Tracker distance threshold (lower = stricter)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Place a YOLOv8 ONNX file there (e.g., yolov8n.onnx)."
        )

    detector = YoloV8OnnxDetector(model_path=model_path, class_names=COCO_CLASSES, conf_threshold=args.confidence, iou_threshold=args.iou)
    tracker = build_tracker(distance_threshold=args.distance_threshold)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = None
    annotated_path = output_dir / "annotated.mp4"
    tracks_csv = output_dir / "tracks.csv"
    log_path = output_dir / "run_log.txt"

    timings: List[FrameTimings] = []
    tracks_rows = []
    processed_frames = 0
    total_seconds = 0.0
    frames_seen = 0

    for frame_idx in tqdm(range(total_frames if total_frames > 0 else 10**9), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        frame = maybe_resize(frame, args.resize)
        frames_seen += 1

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))

        detection_ms = tracking_ms = render_ms = 0.0

        if frame_idx % max(args.every_n_frames, 1) == 0:
            start_det = time.perf_counter()
            detections = detector(frame, args.img_size)
            detection_ms = (time.perf_counter() - start_det) * 1000

            nf_detections = detections_to_norfair(detections)
            start_track = time.perf_counter()
            tracked_objects = tracker.update(nf_detections)
            tracking_ms = (time.perf_counter() - start_track) * 1000
            track_ids = [obj.id for obj in tracked_objects]
            annotated = draw_detections(frame, detections, track_ids)

            for det, track_id in zip(detections, track_ids):
                ts = timestamp_from_frame(frame_idx, fps)
                tracks_rows.append((frame_idx, ts, track_id, det.class_name, det.score, *det.bbox))

            processed_frames += 1
            timings.append(FrameTimings(frame_idx, detection_ms, tracking_ms, render_ms))
        else:
            annotated = frame

        start_write = time.perf_counter()
        writer.write(annotated)
        render_ms = (time.perf_counter() - start_write) * 1000
        if detection_ms or tracking_ms:
            if timings:
                timings[-1].render_ms = render_ms
                total_seconds += (detection_ms + tracking_ms + render_ms) / 1000.0

    cap.release()
    if writer:
        writer.release()

    stats = compute_run_stats(
        timings,
        total_frames=frames_seen,
        processed_frames=processed_frames,
        total_seconds=total_seconds,
    )
    write_tracks_csv(tracks_csv, tracks_rows)
    write_run_log(log_path, stats)
    logging.info("Finished. Annotated video at %s", annotated_path)
    logging.info("Tracks CSV at %s", tracks_csv)
    logging.info("Run log at %s", log_path)


if __name__ == "__main__":
    main()
