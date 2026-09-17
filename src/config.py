"""Chemins et constantes partagés par les scripts du projet.

Un seul endroit où sont définis la racine du repo, la livraison BnF et les dossiers
produits, pour éviter qu'ils divergent entre les scripts.
"""

from __future__ import annotations

from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]

# Livraison BnF — LECTURE SEULE, ne jamais y écrire
LIVRAISON = RACINE / "20250214-segmentation-dataset-iiif"

DATA = RACINE / "data"                       # dérivés produits par prepare_dataset.py
MODELS = RACINE / "models"                   # poids pré-entraînés (entrées)
RUNS = RACINE / "runs"                       # sorties d'entraînement
DATASET_DEFAUT = DATA / "detection-nc1" / "dataset.yaml"


def configurer_ultralytics() -> None:
    """Force Ultralytics à télécharger ses poids dans models/.

    Sans ça, il les dépose dans le répertoire courant et crée un dossier `weights/`
    à la racine, au gré de l'endroit d'où le script est lancé.
    """
    from ultralytics.utils import SETTINGS

    MODELS.mkdir(exist_ok=True)
    if Path(SETTINGS.get("weights_dir", "")) != MODELS:
        SETTINGS.update({"weights_dir": str(MODELS)})
