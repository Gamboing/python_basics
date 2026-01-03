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
