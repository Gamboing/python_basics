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
  --log-level INFO \
  --rois config/rois.json \
  --events-csv runs/events.csv
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
- `--rois`: archivo JSON con polígonos/rectángulos para ROIs de interacción.
- `--events-csv`: ruta para exportar eventos (Approach/Pick/Leave).
- `--approach-seconds`: tiempo mínimo dentro de ROI para registrar Approach.
- `--pick-area-delta`: delta relativa de área (bbox) para inferir Pick sin pose.
- `--enable-pose`: activa estimación de pose (modelo ligero ONNX tipo COCO 17 kp, usando muñecas para Pick).
- `--pose-model`: ruta opcional a un modelo de pose; si no se especifica, se busca uno paralelo al detector.

## Formato de config/rois.json

```json
[
  { "id": "shelf_1", "rect": [100, 200, 400, 500] },
  { "id": "promo_zone", "points": [[50, 50], [200, 80], [180, 220], [60, 240]] }
]
```

## Eventos y limitaciones

- Se exporta `events.csv` con columnas `(track_id, roi_id, event_type, t_start, t_end, duration)`.
- El overlay dibuja las ROIs y muestra el estado del track (Approach/Pick/In).
- **Pick sin pose**: se infiere con heurística de cambio de área de bbox; es menos fiable ante oclusiones, ruido de detección o variaciones de escala. Para mayor precisión, habilitar pose/hand keypoints y reemplazar la heurística.
- **Pick con pose**: se usa un modelo ligero de pose (COCO 17 kp) y se consideran las muñecas (left/right wrist) dentro del ROI con score ≥ conf. Si el modelo es pesado para CPU o falla la inferencia, se desactiva con advertencia y se vuelve a la heurística de bbox.

## Tests

```bash
python -m unittest tests/test_pipeline.py
```

## ETL de productos (Excel → Postgres)

```bash
docker compose up -d
python -m tools.ingest_products_excel --db-url $DATABASE_URL --excel /mnt/data/Muebles.xlsx
```
- Los errores y advertencias se guardan en `outputs/etl_errors_products.csv`.

## Licencia

MIT (ver `LICENSE`).
