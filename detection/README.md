# Illustration Detection

Illustration detection pipeline on Gallica images.

## Structure

```
detection/
├── src/
│   ├── config.py               # Shared configurations and constants
│   ├── prepare_dataset.py      # Dataset preparation for YOLO
│   ├── train.py                # Model training
│   ├── predict.py              # Inference on images
│   └── evaluate.py             # Model validation / evaluation
├── docs/
│   └── modeles-detection.md    # Model specs and hyperparameters
├── data/                       # Formatted datasets (ignored by git)
├── models/                     # Pretrained weights / checkpoints (ignored by git)
└── runs/                       # Training and prediction outputs (ignored by git)
```

## Workflow

Commands are run from the repository root:

```bash
# 1. Prepare dataset
uv run python detection/src/prepare_dataset.py

# 2. Train model
uv run python detection/src/train.py

# 3. Evaluate weights
uv run python detection/src/evaluate.py --weights detection/runs/train/weights/best.pt

# 4. Run inference on images
uv run python detection/src/predict.py --weights detection/runs/train/weights/best.pt --source path/to/image.jpg
```
