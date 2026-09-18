# Illustration detection

Detection of illustrations on digitized Gallica views, as part of the PFEE carried out
with the BnF. Classification of the detected illustrations is a separate scope.

## Structure

```
detection/
├── src/
│   ├── config.py             # shared paths, Ultralytics configuration
│   ├── prepare_dataset.py    # BnF delivery -> clean YOLO dataset
│   ├── train.py              # training
│   ├── predict.py            # inference on images, annotated output
│   └── evaluate.py           # detailed metrics (see docs/)
├── docs/
│   └── modeles-detection.md  # evaluation metrics, model, hyperparameters
│
├── data/                     # generated derived datasets (not versioned)
├── models/                   # pretrained weights, inputs (not versioned)
└── runs/                     # training outputs (not versioned)
```

The BnF delivery lives in `../datasets/20250214-segmentation-dataset-iiif/`
(read-only, not versioned). The last three folders are ignored by git: they are either
large or regenerated from the code.

## Workflow

Commands are run from the repo root (where the uv environment lives).

```bash
# 1. derived dataset: class filtering, deduplication, local paths
uv run python detection/src/prepare_dataset.py

# 2. training (defaults = those of docs/modeles-detection.md)
uv run python detection/src/train.py

# 3. detailed evaluation
uv run python detection/src/evaluate.py --weights detection/runs/yolo26l/weights/best.pt

# 4. inference on one or several images (annotated .jpg in detection/runs/predict/)
uv run python detection/src/predict.py --weights detection/runs/yolo26l/weights/best.pt page.jpg --json
```

`predict.py` prints the score and box of each detection. Like the training corpus, images
are squashed to 800×800; `--letterbox` keeps the native aspect ratio.

## The `.pt` files

Three different natures, not to be confused:

| Location | Nature |
|---|---|
| `models/yolo26l.pt` | COCO pretrained weights — training **input** |
| `models/yolo26n.pt` | Nano model downloaded by Ultralytics for its AMP check at startup |
| `runs/<name>/weights/best.pt` | Weights **produced** by training — this is what gets evaluated |

`src/config.py` forces Ultralytics to drop its downloads into `models/`, otherwise they
end up in the current directory.

## Environment

Python 3.12 via uv, PyTorch CUDA, Ultralytics. Development GPU: RTX 4070 Laptop,
8 GB of VRAM — this is what constrains `batch` and `imgsz`.

```bash
uv sync
```
