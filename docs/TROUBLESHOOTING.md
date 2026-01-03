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
