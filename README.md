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
