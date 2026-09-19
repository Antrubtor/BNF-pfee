"""Paths and constants shared by the project scripts.

Single place where the repo root, the detection/ folder, the BnF delivery and the
generated folders are defined, so they cannot diverge between scripts.
"""

from __future__ import annotations

from pathlib import Path

DETECTION = Path(__file__).resolve().parents[1]
ROOT = DETECTION.parent

# BnF delivery — READ-ONLY, never write into it
DELIVERY = ROOT / "datasets" / "20250214-segmentation-dataset-iiif"

DATA = DETECTION / "data"                    # derived datasets produced by prepare_dataset.py
MODELS = DETECTION / "models"                # pretrained weights (inputs)
RUNS = DETECTION / "runs"                    # training outputs
DEFAULT_DATASET = DATA / "detection-nc1" / "dataset.yaml"


def configure_ultralytics() -> None:
    """Force Ultralytics to download its weights into models/.

    Otherwise it drops them in the current directory and creates a `weights/` folder
    at the root, depending on where the script is launched from.
    """
    from ultralytics.utils import SETTINGS

    MODELS.mkdir(exist_ok=True)
    if Path(SETTINGS.get("weights_dir", "")) != MODELS:
        SETTINGS.update({"weights_dir": str(MODELS)})
