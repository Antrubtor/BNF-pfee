# Détection d'illustrations

Détection des illustrations sur vues numérisées Gallica, dans le cadre du PFEE mené
avec la BnF. La classification des illustrations détectées relève d'un autre périmètre.

## Structure

```
detection/
├── src/
│   ├── config.py             # chemins partagés, configuration Ultralytics
│   ├── prepare_dataset.py    # livraison BnF -> dataset YOLO propre
│   ├── train.py              # entraînement
│   └── evaluate.py           # métriques détaillées (cf. docs/)
├── docs/
│   └── modeles-detection.md  # métriques d'évaluation, modèle, hyperparamètres
│
├── data/                     # dérivés générés (non versionné)
├── models/                   # poids pré-entraînés, entrées (non versionné)
└── runs/                     # sorties d'entraînement (non versionné)
```

La livraison BnF se trouve dans `../datasets/20250214-segmentation-dataset-iiif/`
(lecture seule, non versionnée). Les trois derniers dossiers sont ignorés par git : soit
ils sont volumineux, soit ils se régénèrent à partir du code.

## Enchaînement

Les commandes se lancent depuis la racine du repo (où vit l'environnement uv).

```bash
# 1. dataset dérivé : filtrage des classes, déduplication, chemins locaux
uv run python detection/src/prepare_dataset.py

# 2. entraînement (valeurs par défaut = celles de docs/modeles-detection.md)
uv run python detection/src/train.py

# 3. évaluation détaillée
uv run python detection/src/evaluate.py --weights detection/runs/yolo26l/weights/best.pt
```

## Les `.pt`

Trois natures différentes, à ne pas confondre :

| Emplacement | Nature |
|---|---|
| `models/yolo26l.pt` | Poids pré-entraînés COCO — **entrée** de l'entraînement |
| `models/yolo26n.pt` | Modèle nano téléchargé par Ultralytics pour son test AMP au démarrage |
| `runs/<nom>/weights/best.pt` | Poids **produits** par l'entraînement — c'est ce qu'on évalue |

`src/config.py` force Ultralytics à déposer ses téléchargements dans `models/`, faute de
quoi ils atterrissent dans le répertoire courant.

## Environnement

Python 3.12 via uv, PyTorch CUDA, Ultralytics. GPU de développement : RTX 4070 Laptop,
8 Go de VRAM — c'est ce qui contraint `batch` et `imgsz`.

```bash
uv sync
```
