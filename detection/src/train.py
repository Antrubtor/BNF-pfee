#!/usr/bin/env python3
"""Train an illustration detector on the derived dataset.

Defaults are the values settled in docs/modeles-detection.md.
The dataset must have been produced beforehand by prepare_dataset.py.

Usage:
    uv run python detection/src/train.py                        # full run, 50 epochs
    uv run python detection/src/train.py --epochs 2 --name test # quick smoke test
    uv run python detection/src/train.py --no-mosaic --no-fliplr
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_DATASET, RUNS, configure_ultralytics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26l.pt", help="starting weights")
    p.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=800, help="corpus resolution, multiple of 32")
    p.add_argument("--batch", type=int, default=6, help="6 fits in 8 GB of VRAM (measured)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="yolo26l", help="run name inside runs/")
    p.add_argument("--project", type=Path, default=RUNS)
    # questionable augmentations on documents: see docs/modeles-detection.md
    p.add_argument("--no-mosaic", action="store_true", help="disable mosaic (fake page layouts)")
    p.add_argument("--no-fliplr", action="store_true", help="disable horizontal flip (mirrors text)")
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found — run prepare_dataset.py first")

    configure_ultralytics()
    from ultralytics import YOLO  # late import: faster startup on argument errors

    overrides = {}
    if args.no_mosaic:
        overrides["mosaic"] = 0.0
    if args.no_fliplr:
        overrides["fliplr"] = 0.0

    print(f"model {args.model} · {args.epochs} epochs · imgsz {args.imgsz} · batch {args.batch}")
    if overrides:
        print(f"augmentations disabled: {', '.join(overrides)}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        amp=True,
        exist_ok=True,
        **overrides,
    )

    weights = args.project / args.name / "weights" / "best.pt"
    print(f"\nWeights: {weights}")
    print(f"Detailed evaluation:\n  uv run python detection/src/evaluate.py --weights {weights}")


if __name__ == "__main__":
    main()
