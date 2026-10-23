# Object counting on a real street video — ejecución local del notebook de Ultralytics

Implementación ejecutada del notebook oficial
[`how-to-count-the-objects-using-ultralytics-yolo.ipynb`](../notebooks/how-to-count-the-objects-using-ultralytics-yolo.ipynb)
del repositorio [ultralytics/notebooks](https://github.com/ultralytics/notebooks),
adaptada para correr **fuera de Colab, sin GPU**, sobre un video real descargado de YouTube.

Resultado: [`output/counting_street.mp4`](output/counting_street.mp4) — video MP4
(1920×1080, H.264, 13,4 s) con detección + tracking + conteo por zona dibujado sobre cada frame.

## 1. Insumos

| Elemento | Valor |
|---|---|
| Modelo | `yolo26n.pt` (YOLO26 nano, COCO, 80 clases) |
| Paquete | `ultralytics 8.4.184` |
| Video fuente | *Busy Street With People Crossing Pedestrian Timelapse* — YouTube `mKbIkpHofOM`, tramo completo (13,5 s) |
| Resolución / fps | 1920×1080 @ 30 fps |
| Hardware | CPU only (12 hilos, sin driver NVIDIA disponible en el host) |
| Inferencia | `imgsz=640`, `conf=0.25`, tracker del `solutions.ObjectCounter` |

El video fuente está incluido en [`media/street_nyc_1080p.mp4`](media/street_nyc_1080p.mp4)
(fuente original: YouTube, uso demostrativo).

## 2. Qué cambia respecto del notebook original

El notebook está pensado para Colab (`!uv pip install`, ventana GUI de OpenCV, video
de demo de Ultralytics y una zona de conteo fija para 1080p). La adaptación:

| Notebook original | Esta implementación |
|---|---|
| `!uv pip install ultralytics` | `requirements.txt` + venv (`python3 -m venv --system-site-packages`) |
| `safe_download(...solutions-ci-demo.mp4)` | video real descargado de YouTube con `yt-dlp` |
| `region_points` fijos `(20,400)-(1080,400)` | zona reescalada a la resolución real: banda `(600,720)-(1700,800)` |
| `solutions.ObjectCounter(show=True)` | `show=False` (headless): los frames se escriben a disco |
| `cv2.VideoWriter("counting.avi", mp4v)` | `.avi` intermedio → **MP4 H.264** (`libx264`, `crf 20`, `+faststart`) |
| todas las clases COCO | `classes=[0,1,2,3,5,7]` (persona, bici, auto, moto, bus, camión) |
| sin métricas | `output/stats.json` con tiempos y conteos reales |

La lógica del notebook se conserva: `ObjectCounter` + `region` + loop
`cap.read()` → `counter(im0)` → `video_writer.write(results.plot_im)`.

## 3. Cómo reproducirlo

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python count_objects_street.py \
    --source media/street_nyc_1080p.mp4 --outdir output
```

Para descargar de nuevo el video fuente:

```bash
yt-dlp -f "137+140" --merge-output-format mp4 \
    -o "media/street_nyc_1080p.%(ext)s" "https://www.youtube.com/watch?v=mKbIkpHofOM"
```

## 4. Resultados medidos (`output/stats.json`)

| Métrica | Valor |
|---|---|
| Frames procesados | 402 (13,4 s @ 30 fps) |
| Tiempo total | 49,46 s |
| Latencia media por frame | 99,1 ms |
| Throughput en CPU | 8,13 fps |
| Objetos contados en la zona | 18 (17 IN / 1 OUT) |
| Desglose al final del video | Car IN 12 · Bus IN 2 · Truck OUT 1 → 18 |
| Archivo de salida | `output/counting_street.mp4`, 15,1 MB, H.264 yuv420p |

Detalle del panel en el frame final del video:

```
Car:   IN 12 OUT 0
Bus:   IN 2  OUT 0
Truck: IN 0  OUT 1
```

## 5. Estructura

```
local-demo/
├── count_objects_street.py     # implementación adaptada (equivalente al notebook)
├── requirements.txt
├── media/street_nyc_1080p.mp4  # insumo real (YouTube)
└── output/
    ├── counting_street.mp4     # ENTREGABLE: video anotado
    └── stats.json              # métricas de la ejecución
```

## 6. Notas

- Sin GPU: la RTX/Quadro del host no tiene driver cargado (`nvidia-smi` falla), por eso
  todo corre en CPU a ~8 fps. Con GPU el mismo script procesa el clip en tiempo real.
- El `.avi` intermedio (`counting_street_raw.avi`, 36 MB) no se versiona: se regenera al ejecutar.
- Código y notebooks originales: © Ultralytics, licencia del repositorio upstream.
