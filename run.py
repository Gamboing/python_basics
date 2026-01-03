import argparse
import logging
import os
from pathlib import Path

from src.config import AppConfig, ExportConfig, mode_defaults


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-first ONNX detector pipeline")
    parser.add_argument("--model-path", type=Path, required=True, help="Ruta al modelo ONNX.")
    parser.add_argument("--source", type=str, default="0", help="Ruta a video/imagen o webcam id.")
    parser.add_argument("--output", type=Path, default=None, help="Ruta opcional para video de salida.")
    parser.add_argument("--events-csv", type=Path, default=None, help="Archivo de eventos (CSV).")
    parser.add_argument("--every-n-frames", type=int, default=None, help="Procesar cada n frames (default por modo).")
    parser.add_argument("--imgsz", type=int, default=None, help="Tamaño de entrada cuadrado.")
    parser.add_argument("--conf", type=float, default=None, help="Umbral de confianza.")
    parser.add_argument("--iou", type=float, default=None, help="IoU para NMS.")
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast", help="Perfil de rendimiento.")
    parser.add_argument("--save-json", type=Path, default=None, help="Guardar detecciones/tracks en JSON.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Guardar detecciones/tracks en CSV.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limitar número de frames procesados.")
    parser.add_argument("--dry-run", action="store_true", help="Procesa solo 100 frames para prueba rápida.")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (DEBUG, INFO, WARNING...).")
    parser.add_argument("--rois", type=Path, default=None, help="Ruta a config/rois.json")
    parser.add_argument("--approach-seconds", type=float, default=1.0, help="Tiempo mínimo dentro de ROI para 'Approach'.")
    parser.add_argument("--pick-area-delta", type=float, default=0.2, help="Delta relativa de área bbox para inferir 'Pick' sin pose.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    defaults = mode_defaults(args.mode)
    max_frames = args.max_frames if args.max_frames is not None else (100 if args.dry_run else None)
    return AppConfig(
        mode=args.mode,
        model_path=args.model_path,
        source=args.source,
        output=args.output,
        max_frames=max_frames,
        dry_run=args.dry_run,
        rois_path=args.rois,
        approach_seconds=args.approach_seconds,
        pick_area_delta=args.pick_area_delta,
        video=defaults.video.override(imgsz=args.imgsz, every_n_frames=args.every_n_frames),
        detector=defaults.detector.override(conf=args.conf, iou=args.iou, imgsz=args.imgsz),
        tracker=defaults.tracker,
        export=ExportConfig(json_path=args.save_json, csv_path=args.save_csv, events_path=args.events_csv),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {args.model_path}")
    if not args.model_path.is_file():
        raise IsADirectoryError(f"Ruta de modelo no es archivo: {args.model_path}")
    if not os.access(args.model_path, os.R_OK):
        raise PermissionError(f"Sin permisos de lectura para {args.model_path}")
    if args.rois and not args.rois.exists():
        raise FileNotFoundError(f"Archivo de ROIs no encontrado en {args.rois}")
    config = build_config(args)
    if config.output:
        config.output.parent.mkdir(parents=True, exist_ok=True)
    from src.pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
