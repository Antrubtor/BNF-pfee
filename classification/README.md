# BnF - Classification Readme

Simple pipeline to classify images using a ConvNeXt model.

## File Structure

- `main.py` : Main entry point that runs all steps in order.
- `dataset.py` : Loads CSVs, prints number of rows, and creates DataLoaders.
- `model.py` : Initializes ConvNeXt model.
- `train.py` : Trains for 2 epochs, times the total training time, and saves the weights.
- `evaluate.py` : Tests on the validation set and writes results to a `.txt` file.

## Installation

```bash
uv sync
```

Do not forget to activate the virtual environment before running the code:

```bash
source .venv/bin/activate # or .venv\Scripts\activate on Windows
```

## How to Run

### 1. Run full training (full dataset)

```bash
uv run python classification/main.py
```

### 2. Quick test (runs 2 epochs **on a small sample**)

```bash
uv run python classification/main.py fast
```
