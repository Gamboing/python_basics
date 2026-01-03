from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoConfig:
    source: str = "0"
    imgsz: int = 640
    every_n_frames: int = 2

    def override(self, *, imgsz: Optional[int] = None, every_n_frames: Optional[int] = None) -> "VideoConfig":
        return replace(
            self,
            imgsz=imgsz if imgsz is not None else self.imgsz,
            every_n_frames=every_n_frames if every_n_frames is not None else self.every_n_frames,
        )


@dataclass(frozen=True)
class DetectorConfig:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640

    def override(self, *, conf: Optional[float] = None, iou: Optional[float] = None, imgsz: Optional[int] = None) -> "DetectorConfig":
        return replace(
            self,
            conf=conf if conf is not None else self.conf,
            iou=iou if iou is not None else self.iou,
            imgsz=imgsz if imgsz is not None else self.imgsz,
        )


@dataclass(frozen=True)
class TrackerConfig:
    max_missed: int = 30
    min_hits: int = 1
    iou_match: float = 0.3


@dataclass(frozen=True)
class ExportConfig:
    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None


@dataclass(frozen=True)
class AppConfig:
    mode: str
    model_path: Path
    source: str
    output: Optional[Path]
    max_frames: Optional[int]
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def mode_defaults(mode: str) -> "AppConfig":
    if mode == "quality":
        video = VideoConfig(imgsz=896, every_n_frames=1)
        detector = DetectorConfig(conf=0.28, iou=0.5, imgsz=896)
    else:  # fast
        video = VideoConfig(imgsz=640, every_n_frames=2)
        detector = DetectorConfig(conf=0.23, iou=0.45, imgsz=640)

    return AppConfig(
        mode=mode,
        model_path=Path("model.onnx"),
        source="0",
        output=None,
        max_frames=None,
        video=video,
        detector=detector,
        tracker=TrackerConfig(),
        export=ExportConfig(),
    )
