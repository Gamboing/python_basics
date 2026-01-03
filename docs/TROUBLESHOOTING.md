# Solución de problemas

## ONNXRuntime no encuentra el modelo
- Verifica la ruta de `--model-path`.
- Comprueba permisos de lectura.
- Asegúrate de que el archivo sea ONNX exportado con input `[1,3,H,W]`.

## El video no abre
- Usa `--source 0` para webcam o ruta absoluta para archivos.
- Confirma códecs instalados; en Linux, instala `ffmpeg` si usas OpenCV precompilado.

## Latencia alta en CPU
- Cambia a `--mode fast`.
- Incrementa `--every-n-frames` a 3 o 4.
- Reduce `--imgsz` a 512.
- Usa entradas de menor resolución de origen.

## Salida vacía o pocas detecciones
- Baja `--conf` a 0.2.
- Revisa que el modelo espere normalización `[0,1]` y orden RGB.
- Aumenta `--imgsz` (modo quality) si la escena tiene objetos pequeños.

## Archivos de exportación no aparecen
- Incluye `--save-json` o `--save-csv`.
- Comprueba que las carpetas padre existan; el pipeline las creará si es posible.
# Troubleshooting

## 1) El video no abre o FPS = 0
- **Síntoma:** `Could not open video` o FPS reportado como 0.
- **Causa:** Códec no soportado por OpenCV.
- **Fix:** Instala FFmpeg y re-encodea: `ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset fast -c:a copy output.mp4`.

## 2) Modelo no encontrado
- **Síntoma:** `Model not found: models/yolov8n.onnx`.
- **Fix:** Descarga el ONNX de YOLOv8n una vez con internet y colócalo en `models/yolov8n.onnx`.

## 3) onnxruntime falla a cargar
- **Síntoma:** Error de carga o falta de DLLs.
- **Fix:** Reinstala `onnxruntime` (`pip install --force-reinstall onnxruntime`), asegúrate de usar Python de 64 bits.

## 4) Bajo rendimiento (CPU)
- **Flags útiles:**
  - `--resize 960` o `--resize 1280` para bajar la resolución.
  - `--every-n-frames 2` o `3` para saltar frames.
  - `--confidence 0.35` para reducir falsos positivos y carga de NMS.
- **Hardware:** Cierra otras apps; usar SSD mejora IO.

## 5) Norfair produce IDs inestables
- **Fix:**
  - Reduce `--distance-threshold` (ej. 0.6) para matching más estricto.
  - Evita `--every-n-frames` muy alto si los objetos se mueven rápido.

## 6) Salida vacía en tracks.csv
- **Causas:** Umbral de confianza muy alto o la clase buscada no está en el modelo.
- **Fix:** Baja `--confidence` a 0.25-0.3. Confirma que el modelo incluye la clase (COCO por defecto).

## 7) Colores o texto ilegibles en annotated.mp4
- **Fix:** Ajusta `draw_detections` en `src/video_utils.py` (color, grosor, tamaño de fuente) o usa `--resize` para agrandar cajas.

## 8) Ejecutar offline
- Una vez descargado el ONNX y las dependencias (pip), el pipeline corre sin internet.
