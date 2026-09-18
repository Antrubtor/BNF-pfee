#!/usr/bin/env python3
"""Run illustration detection on one or several images.

By default, images are squashed to imgsz x imgsz without preserving the aspect ratio,
exactly like the training corpus (see CLAUDE.md, "800x800 anisotropic resize"). Use
--letterbox to keep the native aspect ratio instead (Ultralytics default behaviour).

Outputs, in --out (default: detection/runs/predict/):
  - <name>.jpg   the image with the detected boxes drawn
  - <name>.json  boxes as normalized xyxy + score (with --json)

Usage:
    uv run python detection/src/predict.py --weights detection/runs/yolo26l/weights/best.pt page.jpg
    uv run python detection/src/predict.py --weights ... images_dir/ --conf 0.4 --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from config import RUNS, configure_ultralytics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def collect_images(inputs: list[Path]) -> list[Path]:
    """Expand directories into their image files, keep explicit files as they are."""
    images = []
    for path in inputs:
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
        else:
            raise SystemExit(f"{path} not found")
    if not images:
        raise SystemExit("no image to process")
    return images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+", help="image files and/or directories")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou-nms", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--nms", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-nms for YOLO26 end-to-end inference")
    p.add_argument("--letterbox", action="store_true",
                   help="keep the native aspect ratio instead of squashing to imgsz x imgsz")
    p.add_argument("--out", type=Path, default=RUNS / "predict", help="output folder")
    p.add_argument("--json", action="store_true", help="also write the boxes as JSON")
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found")
    images = collect_images(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)

    configure_ultralytics()
    from ultralytics import YOLO  # late import: faster startup on argument errors

    model = YOLO(args.weights)

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path.name}: unreadable, skipped")
            continue
        if not args.letterbox:
            frame = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou_nms,
                               nms=args.nms, verbose=False)[0]
        h, w = result.orig_shape
        boxes = [
            {"score": round(float(s), 4), "xyxy": [round(float(v), 6) for v in b]}
            for b, s in zip((result.boxes.xyxy.cpu().numpy() / [w, h, w, h]).tolist(),
                            result.boxes.conf.cpu().numpy())
        ]

        cv2.imwrite(str(args.out / f"{path.stem}.jpg"), result.plot())
        if args.json:
            (args.out / f"{path.stem}.json").write_text(json.dumps(
                {"image": str(path), "conf": args.conf, "boxes": boxes}, indent=2))
        print(f"{path.name}: {len(boxes)} illustration(s)")
        for i, box in enumerate(sorted(boxes, key=lambda b: -b["score"]), 1):
            x1, y1, x2, y2 = box["xyxy"]
            print(f"  #{i}  score {box['score']:.3f}  "
                  f"xyxy ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})  "
                  f"area {(x2 - x1) * (y2 - y1):.1%} of page")

    print(f"\nOutputs written to {args.out}")


if __name__ == "__main__":
    main()
