from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2

from src.config import AppConfig
from src.detector_onnx import OnnxDetector
from src.exporters import ExportBuffer, write_csv, write_json
from src.tracker import IoUTracker
from src.video_io import VideoWriter, iter_frames


class Pipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = OnnxDetector(str(config.model_path), config.detector)
        self.tracker = IoUTracker(config.tracker)
        self.export_buffer = ExportBuffer()
        self.writer = None

    def _init_writer(self, frame_shape) -> None:
        if not self.config.output:
            return
        height, width = frame_shape[:2]
        fps = 30.0  # fallback si no se conoce la fuente
        self.writer = VideoWriter(self.config.output, fps=fps, frame_size=(width, height))

    def _draw(self, frame, tracks) -> None:
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"ID {track.track_id} {track.score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )

    def _export_frame(self, frame_idx: int, timestamp_ms, tracks) -> None:
        for track in tracks:
            record: Dict[str, object] = {
                "frame": frame_idx,
                "time_ms": timestamp_ms,
                "track_id": track.track_id,
                "score": track.score,
                "cls": track.cls,
                "bbox": track.bbox,
            }
            self.export_buffer.rows.append(record)

    def run(self) -> None:
        for frame_data in iter_frames(
            self.config.source,
            every_n=self.config.video.every_n_frames,
            max_frames=self.config.max_frames,
        ):
            if self.writer is None and self.config.output:
                self._init_writer(frame_data.image.shape)

            detections = self.detector(frame_data.image)
            tracks = self.tracker.update(detections)
            self._draw(frame_data.image, tracks)
            self._export_frame(frame_data.index, frame_data.timestamp_ms, tracks)
            if self.writer:
                self.writer.write(frame_data.image)

        if self.writer:
            self.writer.close()
        self._flush_exports()

    def _flush_exports(self) -> None:
        if self.config.export.json_path:
            self._ensure_parent(self.config.export.json_path)
            write_json(self.config.export.json_path, self.export_buffer.rows)
        if self.config.export.csv_path:
            self._ensure_parent(self.config.export.csv_path)
            write_csv(self.config.export.csv_path, self.export_buffer.rows)

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
