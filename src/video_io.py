from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2


@dataclass
class FrameData:
    index: int
    image: "cv2.Mat"
    timestamp_ms: Optional[float]


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")
    return cap


def iter_frames(source: str, every_n: int = 1, max_frames: Optional[int] = None) -> Generator[FrameData, None, None]:
    cap = open_capture(source)
    try:
        idx = 0
        yielded = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                yield FrameData(index=idx, image=frame, timestamp_ms=timestamp_ms if timestamp_ms > 0 else None)
                yielded += 1
                if max_frames and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, path: Path, fps: float, frame_size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"No se pudo abrir escritor de video en {path}")

    def write(self, frame: "cv2.Mat") -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
