# CPU-first ONNX detection pipeline

Pequeño proyecto de referencia para detección ONNX en CPU con modos **fast** y **quality**, lectura de video/imágenes y exportación de resultados.

## Requisitos

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

```bash
python run.py \
  --model-path models/detector.onnx \
  --source data/video.mp4 \
  --mode fast \
  --output runs/out.mp4 \
  --log-level INFO
  --output runs/out.mp4
```

## Modos y parámetros recomendados

- **imgsz/resize**: 640×640 (modo fast: 512–640; modo quality: 768–960).
- **conf**: 0.25 por defecto (fast: 0.20–0.25; quality: 0.25–0.30).
- **iou**: 0.45 por defecto (fast/quality: 0.45–0.50).
- **every-n-frames**: 2 por defecto (fast: 3–4; quality: 1–2).

## Estructura

- `run.py`: punto de entrada CLI.
- `src/config.py`: dataclasses de configuración y defaults por modo.
- `src/video_io.py`: carga de videos, lectura de frames y escritor opcional.
- `src/detector_onnx.py`: envoltura de inferencia ONNXRuntime + NMS.
- `src/tracker.py`: rastreador simple estilo SORT basado en IoU.
- `src/pipeline.py`: orquestación de lectura → detección → tracking → export.
- `src/exporters.py`: exportación a JSON/CSV de detecciones y tracks.
- `docs/TROUBLESHOOTING.md`: resolución de problemas comunes.

## Ejemplo avanzado

```bash
```bas
python run.py \
  --model-path models/detector.onnx \
  --source data/video.mp4 \
  --every-n-frames 3 \
  --mode quality \
  --output runs/out.mp4 \
  --save-json runs/out.json \
  --dry-run
```

## Flags útiles de robustez

- `--dry-run`: procesa 100 frames para probar rápido la ruta de datos y el modelo.
- `--log-level`: controla verbosidad (DEBUG/INFO/WARNING) e imprime tiempos por etapa y FPS aproximado.

## Tests

```bash
python -m unittest tests/test_pipeline.py
```

## Licencia

MIT (ver `LICENSE`).
  --save-json runs/out.json
```

## Licencia

MIT (ver `LICENSE`).
# CPU-only YOLOv8 ONNX + Norfair Tracking (Windows-friendly)

Proceso completo para cargar un video local, detectar personas/productos con YOLOv8n (ONNX, CPU) y trackear con Norfair. Genera:
- `outputs/annotated.mp4`: video con cajas, clase, confianza e `id` temporal.
- `outputs/tracks.csv`: columnas `frame, timestamp_sec, track_id, class_name, conf, x1, y1, x2, y2`.
- `outputs/run_log.txt`: FPS promedio y tiempos por etapa.

## Requisitos previos
- Windows 11 (probado en CPU; sin GPU).
- Python 3.10+ (ejemplo: 3.12).
- Video local en formato MP4/H.264 u otro codec soportado por OpenCV.
- Modelo YOLOv8n exportado a ONNX (colócalo en `models/yolov8n.onnx`).
  - Descarga una vez desde un equipo con internet: `https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx`
  - Copia el archivo al directorio `models/` de este proyecto.

## Instalación
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
> Tip: En Linux/macOS, activa con `source .venv/bin/activate`.

## Uso básico
```bash
python run.py --video-path "RUTA/A/TU_VIDEO.mp4" --model-path models/yolov8n.onnx --output-dir outputs \
  --resize 1280 --every-n-frames 1 --img-size 640 --confidence 0.3 --iou 0.5 --distance-threshold 0.7
```
Parámetros clave (CPU-friendly):
- `--resize`: ancho objetivo para reducir resolución (ej. 1280 o 960). Mantiene aspecto.
- `--every-n-frames`: procesa cada N frames (ej. 2 o 3) para mayor FPS.
- `--img-size`: lado cuadrado de entrada del modelo (640 recomendado para YOLOv8n).

Salidas se guardan en `outputs/` (creado automáticamente). Corre 100% offline una vez tengas el modelo y dependencias instaladas.

## Estructura
```
python_basics/
├─ run.py                # Script principal CLI
├─ requirements.txt      # Dependencias mínimas (CPU)
├─ src/
│  ├─ detector.py        # Wrapper YOLOv8 ONNX + NMS
│  ├─ tracking.py        # Tracker Norfair (IoU distance)
│  └─ video_utils.py     # Helpers de video, dibujo, logging
├─ docs/
│  └─ TROUBLESHOOTING.md # Errores comunes y soluciones
└─ outputs/              # (Se crea en runtime) video, csv, log
```

## Notas de rendimiento (solo CPU)
- Usa `--resize` y `--every-n-frames` para ajustar precisión vs velocidad.
- Prefiere `yolov8n` (nano) para CPU. Modelos más grandes bajarán el FPS.
- Si tu video es 1080p/4K, reducir a 960-1280 suele mejorar mucho la velocidad.

## Limitaciones
- IDs son temporales (sin identidad real ni reconocimiento facial).
- Solo detección + tracking; no incluye eventos (approach/pick/leave) ni conteo por zona.

## Ejemplo mínimo
```bash
python run.py --video-path "C:\\videos\\demo.mp4" --model-path models/yolov8n.onnx --resize 1280 --every-n-frames 2
```
