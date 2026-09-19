# Illustration Detection — Model & Specs

## Scope

Detect illustration regions on scanned Gallica pages.
The detected bounding boxes are intended for afterwards classification.

## Dataset used

- **Class**: The model only detects a single class that we called `Illustration`.
- **Text annotations**: For now ignored (treated as background).
- **Source**: `datasets/20250214-segmentation-dataset-iiif/`.

## Model used

- **Architecture**: YOLO (Ultralytics)
- **Dataset used**
- **Base Resolution**: 800
- **Epochs**: 50
- **Classes**: 1

## Evaluation

Model evaluation uses standard object detection metrics:

- **mAP50**: Mean Average Precision at IoU 0.50
- **mAP50-95**: Mean Average Precision averaged over IoU 0.50 to 0.95
- **Precision / Recall**: Operating point metrics
