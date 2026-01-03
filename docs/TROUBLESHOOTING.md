# Solución de problemas (10 casos comunes)

1) **ONNXRuntime no encuentra el modelo**  
   - Verifica la ruta de `--model-path`.  
   - Comprueba permisos de lectura.  
   - El archivo debe ser ONNX exportado con input `[1,3,H,W]`.

2) **`Permission denied` al abrir el modelo**  
   - Otorga permisos de lectura al archivo (`chmod +r`).  
   - Evita rutas sobre FS montados solo lectura.

3) **El video no abre**  
   - Usa `--source 0` para webcam o ruta absoluta para archivos.  
   - Confirma códecs instalados; en Linux, instala `ffmpeg` si usas OpenCV precompilado.

4) **`Permission denied` al leer la fuente**  
   - Comprueba permisos en la carpeta/archivo de video.  
   - Si está en red, copia localmente para descartar restricciones.

5) **No se puede escribir el video de salida**  
   - Verifica permisos de escritura en la carpeta de destino.  
   - Cambia el contenedor a `.mp4` y confirma que el codec `mp4v` esté disponible.

6) **Latencia alta en CPU**  
   - Cambia a `--mode fast`.  
   - Incrementa `--every-n-frames` a 3 o 4.  
   - Reduce `--imgsz` a 512 o menor.  
   - Usa entradas de menor resolución de origen.

7) **Salida vacía o pocas detecciones**  
   - Baja `--conf` a 0.2.  
   - Revisa que el modelo espere normalización `[0,1]` y orden RGB.  
   - Aumenta `--imgsz` (modo quality) si la escena tiene objetos pequeños.

8) **Errores de tracking**  
   - Si el tracker lanza excepción, el pipeline continúa solo con detección.  
   - Revisa los datos de entrada (bbox válidos) y ajusta `tracker.iou_match`.

9) **Exportaciones CSV/JSON vacías**  
   - Asegúrate de pasar `--save-json` o `--save-csv`.  
   - Verifica que haya detecciones; de otro modo los archivos estarán vacíos.

10) **Incompatibilidades de ONNX (opset)**  
    - Re-exporta el modelo con opset ≥13 para ONNXRuntime 1.17.  
    - Revisa errores en consola; si faltan operadores, incluye implementaciones o cambia a una versión soportada.

11) **ROIs no cargan o eventos vacíos**  
    - Confirma que `--rois` apunta a un JSON válido con `id` + `points` o `rect`.  
    - Revisa el log: si hay excepciones al cargar, el pipeline continúa sin ROIs.  
    - Si no hay tracks dentro de ROIs, `events.csv` puede quedar vacío o con pocas filas.
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
