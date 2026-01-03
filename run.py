import argparse
import os
from pathlib import Path

from src.config import AppConfig, ExportConfig, mode_defaults
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-first ONNX detector pipeline")
    parser.add_argument("--model-path", type=Path, required=True, help="Ruta al modelo ONNX.")
    parser.add_argument("--source", type=str, default="0", help="Ruta a video/imagen o webcam id.")
    parser.add_argument("--output", type=Path, default=None, help="Ruta opcional para video de salida.")
    parser.add_argument("--every-n-frames", type=int, default=None, help="Procesar cada n frames (default por modo).")
    parser.add_argument("--imgsz", type=int, default=None, help="Tamaño de entrada cuadrado.")
    parser.add_argument("--conf", type=float, default=None, help="Umbral de confianza.")
    parser.add_argument("--iou", type=float, default=None, help="IoU para NMS.")
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast", help="Perfil de rendimiento.")
    parser.add_argument("--save-json", type=Path, default=None, help="Guardar detecciones/tracks en JSON.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Guardar detecciones/tracks en CSV.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limitar número de frames procesados.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    defaults = mode_defaults(args.mode)
    return AppConfig(
        mode=args.mode,
        model_path=args.model_path,
        source=args.source,
        output=args.output,
        max_frames=args.max_frames,
        video=defaults.video.override(imgsz=args.imgsz, every_n_frames=args.every_n_frames),
        detector=defaults.detector.override(conf=args.conf, iou=args.iou, imgsz=args.imgsz),
        tracker=defaults.tracker,
        export=ExportConfig(json_path=args.save_json, csv_path=args.save_csv),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    os.makedirs(config.output.parent, exist_ok=True) if config.output else None
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
