#!/usr/bin/env python3
"""Object counting on a real street video (local runs, CPU only).

Adaptation of the official Ultralytics notebook:
    notebooks/how-to-count-the-objects-using-ultralytics-yolo.ipynb
    https://github.com/ultralytics/notebooks

Original notebook          -> this script
---------------------------------------------------------------
Colab `!uv pip install`    -> requirements.txt + venv (no Colab needed)
solutions-ci-demo.mp4      -> media/street_nyc_1080p.mp4 (YouTube source)
region (20,400)-(1080,400)-> region scaled to the real 1920x1080 frame
`show=True` (GUI window)   -> show=False, frames are written to disk
counting.avi (mp4v)        -> counting_street_raw.avi -> H.264 MP4 via ffmpeg

Usage:
    python count_objects_street.py --source media/street_nyc_1080p.mp4 \
                                   --outdir output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2

from ultralytics import __version__ as ultra_version
from ultralytics import solutions

# Reference frame the region was designed for (full-HD), scaled at runtime.
REF_W, REF_H = 1920, 1080
# Horizontal band on the clear traffic lane of the lower half of the frame.
REGION_REF = [(600, 720), (1700, 720), (1700, 800), (600, 800)]
# COCO ids: person, bicycle, car, motorcycle, bus, truck
CLASSES = [0, 1, 2, 3, 5, 7]


def scale_region(width: int, height: int) -> list[tuple[int, int]]:
    sx, sy = width / REF_W, height / REF_H
    return [(int(x * sx), int(y * sy)) for x, y in REGION_REF]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="media/street_nyc_1080p.mp4")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Error reading video file: {args.source}")
    w, h, fps = (int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    region = scale_region(w, h)

    raw_avi = outdir / "counting_street_raw.avi"
    writer = cv2.VideoWriter(str(raw_avi), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        show=False,                 # headless: no GUI window in this run
        region=region,              # region adapted to the real resolution
        model=args.model,
        classes=CLASSES,            # only people + vehicles, not every COCO class
        conf=args.conf,
        verbose=False,
    )

    print(f"ultralytics {ultra_version} | source={args.source} {w}x{h} @ {fps:g} fps "
          f"| frames={total} | region={region}")

    processed, t_start, per_frame = 0, time.perf_counter(), []
    while cap.isOpened():
        ok, im0 = cap.read()
        if not ok:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        t0 = time.perf_counter()
        counts = counter(im0)          # count the objects in the region
        per_frame.append(time.perf_counter() - t0)
        writer.write(counts.plot_im)   # write the annotated frame
        processed += 1
        if processed % 30 == 0:
            print(f"  frame {processed}/{total} "
                  f"in={counts.in_count} out={counts.out_count}")

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    mean_ms = 1000 * sum(per_frame) / max(len(per_frame), 1)
    stats = {
        "ultralytics": ultra_version,
        "model": args.model,
        "source": str(args.source),
        "frame_size": [w, h],
        "source_fps": fps,
        "frames_processed": processed,
        "region": region,
        "classes": CLASSES,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "in_count": int(counter.in_count),
        "out_count": int(counter.out_count),
        "total_counted": int(counter.in_count + counter.out_count),
        "processing_seconds": round(elapsed, 2),
        "mean_frame_ms": round(mean_ms, 2),
        "processing_fps": round(processed / elapsed, 2),
    }

    mp4 = outdir / "counting_street.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(raw_avi),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
        check=True)
    stats["output_mp4"] = str(mp4)
    stats["output_mb"] = round(mp4.stat().st_size / 1e6, 2)
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
