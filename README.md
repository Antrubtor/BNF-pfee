<div align="center">
  <h1>BnF: Image Detection and Classification for Gallica</h1>

  <img src="https://www.bnf.fr/sites/default/files/2024-02/capture_2.jpg" alt="Gallica" width="800">
</div>

Design of a Computer Vision AI pipeline to automatically detect, segment, and classify heritage illustrations from Gallica (BnF).

```
.
├── datasets/           # datasets used for training and evaluation
├── classification/     # classification of illustrations (cf. classification/README.md)
├── detection/          # détection des illustrations (cf. detection/README.md)
├── pyproject.toml      # configuration file for the Python project
└── uv.lock             # lock file for the uv environment
```

## Installation

```bash
uv sync
```
