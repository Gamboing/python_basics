from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from typing import Dict, List

import cv2

from src.config import AppConfig
from src.detector_onnx import OnnxDetector
from src.exporters import ExportBuffer, write_csv, write_json
from src.tracker import IoUTracker
from src.rois import ROI, load_rois
from src.video_io import VideoWriter, iter_frames


class Pipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = OnnxDetector(str(config.model_path), config.detector)
        self.tracker = IoUTracker(config.tracker)
        self.export_buffer = ExportBuffer()
        self.writer = None
        self._tracking_enabled = True
        self.rois: List[ROI] = self._load_rois()
        self._interactions: Dict[Tuple[int, str], Dict[str, float | bool]] = {}

    def _init_writer(self, frame_shape, fps: float | None) -> None:
        if not self.config.output:
            return
        height, width = frame_shape[:2]
        self.writer = VideoWriter(self.config.output, fps=fps or 30.0, frame_size=(width, height))

    def _draw(self, frame, tracks) -> None:
        # Dibujar ROIs
        for roi in self.rois:
            pts = roi.as_int_points
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (255, 180, 0), 2)
            cv2.putText(frame, f"ROI {roi.roi_id}", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            labels = self._track_labels(track.track_id)
            label_text = " | ".join(labels) if labels else f"ID {track.track_id} {track.score:.2f}"
            cv2.putText(
                frame,
                label_text,
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
        logging.info("Inicio de pipeline | modo=%s | dry_run=%s", self.config.mode, self.config.dry_run)
        start_time = time.perf_counter()
        processed = 0
        for frame_data in iter_frames(
            self.config.source,
            every_n=self.config.video.every_n_frames,
            max_frames=self.config.max_frames,
        ):
            if self.writer is None and self.config.output:
                self._init_writer(frame_data.image.shape, frame_data.fps)

            frame_time = self._frame_time(frame_data)
            t0 = time.perf_counter()
            detections = self.detector(frame_data.image)
            t1 = time.perf_counter()
            if self._tracking_enabled:
                try:
                    tracks = self.tracker.update(detections)
                except Exception:
                    logging.exception("Fallo del tracker; continuando sin tracking.")
                    self._tracking_enabled = False
                    tracks = []
            else:
                tracks = []
            t2 = time.perf_counter()
            self._draw(frame_data.image, tracks)
            t3 = time.perf_counter()
            self._export_frame(frame_data.index, frame_data.timestamp_ms, tracks)
            t4 = time.perf_counter()
            if self.writer:
                try:
                    self.writer.write(frame_data.image)
                except Exception:
                    logging.exception("Error al escribir frame en video de salida; se desactiva escritura.")
                    self.writer = None
            processed += 1
            self._update_interactions(tracks, frame_time)
            logging.info(
                "Frame %s | det=%.2f ms | track=%.2f ms | draw=%.2f ms | export=%.2f ms",
                frame_data.index,
                (t1 - t0) * 1e3,
                (t2 - t1) * 1e3,
                (t3 - t2) * 1e3,
                (t4 - t3) * 1e3,
            )
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
        total = time.perf_counter() - start_time
        fps = processed / total if total > 0 else 0
        logging.info("Fin de pipeline | frames=%s | tiempo=%.2fs | fps≈%.2f", processed, total, fps)

    def _flush_exports(self) -> None:
        if self.config.export.json_path:
            self._ensure_parent(self.config.export.json_path)
            write_json(self.config.export.json_path, self.export_buffer.rows)
        if self.config.export.csv_path:
            self._ensure_parent(self.config.export.csv_path)
            write_csv(self.config.export.csv_path, self.export_buffer.rows)
        if self.config.export.events_path:
            self._ensure_parent(self.config.export.events_path)
            write_csv(self.config.export.events_path, self.export_buffer.events)

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_rois(self) -> List[ROI]:
        if not self.config.rois_path:
            return []
        try:
            return load_rois(self.config.rois_path)
        except Exception:
            logging.exception("No se pudieron cargar ROIs; continuando sin ellos.")
            return []

    def _frame_time(self, frame_data) -> float:
        if frame_data.timestamp_ms is not None:
            return frame_data.timestamp_ms / 1000.0
        fps = frame_data.fps or 30.0
        return frame_data.index / fps

    def _update_interactions(self, tracks, t: float) -> None:
        if not self.rois:
            return
        for track in tracks:
            cx = (track.bbox[0] + track.bbox[2]) / 2
            cy = (track.bbox[1] + track.bbox[3]) / 2
            area = max((track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1]), 1e-6)
            prev_area = None
            if track.history:
                hbox = track.history[-1]
                prev_area = max((hbox[2] - hbox[0]) * (hbox[3] - hbox[1]), 1e-6)
            for roi in self.rois:
                inside = roi.contains(cx, cy)
                key = (track.track_id, roi.roi_id)
                state = self._interactions.get(
                    key,
                    {"inside": False, "enter_time": None, "approach_start": None, "pick_time": None},
                )
                if inside:
                    if not state["inside"]:
                        state = {
                            "inside": True,
                            "enter_time": t,
                            "approach_start": None,
                            "pick_time": None,
                        }
                    if state["enter_time"] is not None and state["approach_start"] is None and t - state["enter_time"] >= self.config.approach_seconds:
                        state["approach_start"] = state["enter_time"]
                    if state["pick_time"] is None and prev_area:
                        delta = abs(area - prev_area) / prev_area
                        if delta >= self.config.pick_area_delta and state["enter_time"] is not None:
                            state["pick_time"] = t
                else:
                    if state["inside"]:
                        leave_time = t
                        if state["approach_start"] is not None:
                            self._record_event(track.track_id, roi.roi_id, "Approach", state["approach_start"], leave_time)
                        if state["pick_time"] is not None:
                            self._record_event(track.track_id, roi.roi_id, "Pick", state["pick_time"], leave_time)
                        self._record_event(track.track_id, roi.roi_id, "Leave", leave_time, leave_time)
                    state = {"inside": False, "enter_time": None, "approach_start": None, "pick_time": None}
                self._interactions[key] = state

    def _record_event(self, track_id: int, roi_id: str, event: str, start: float, end: float) -> None:
        duration = max(0.0, end - start)
        self.export_buffer.events.append(
            {
                "track_id": track_id,
                "roi_id": roi_id,
                "event_type": event,
                "t_start": start,
                "t_end": end,
                "duration": duration,
            }
        )

    def _track_labels(self, track_id: int) -> List[str]:
        labels = [f"ID {track_id}"]
        for (tid, roi_id), state in self._interactions.items():
            if tid != track_id:
                continue
            if state.get("inside"):
                if state.get("pick_time") is not None:
                    labels.append(f"Pick@{roi_id}")
                elif state.get("approach_start") is not None:
                    labels.append(f"Approach@{roi_id}")
                else:
                    labels.append(f"In@{roi_id}")
        return labels
