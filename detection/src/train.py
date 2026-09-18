#!/usr/bin/env python3
"""Entraîne un détecteur d'illustrations sur le dataset dérivé.

Les valeurs par défaut sont celles arrêtées dans docs/modeles-detection.md.
Le dataset doit avoir été produit au préalable par prepare_dataset.py.

Usage :
    uv run python detection/src/train.py                        # run complet, 50 epochs
    uv run python detection/src/train.py --epochs 2 --name test # essai rapide
    uv run python detection/src/train.py --no-mosaic --no-fliplr
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DATASET_DEFAUT, RUNS, configurer_ultralytics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26l.pt", help="poids de départ")
    p.add_argument("--data", type=Path, default=DATASET_DEFAUT)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=800, help="résolution du corpus, multiple de 32")
    p.add_argument("--batch", type=int, default=6, help="6 tient dans 8 Go de VRAM (mesuré)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="yolo26l", help="nom du run dans runs/")
    p.add_argument("--project", type=Path, default=RUNS)
    # augmentations douteuses sur du document : cf. docs/modeles-detection.md
    p.add_argument("--no-mosaic", action="store_true", help="désactive mosaic (mises en page factices)")
    p.add_argument("--no-fliplr", action="store_true", help="désactive le miroir horizontal (retourne le texte)")
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} introuvable — lancer d'abord prepare_dataset.py")

    configurer_ultralytics()
    from ultralytics import YOLO  # import tardif : démarrage plus rapide en cas d'erreur d'arguments

    surcharges = {}
    if args.no_mosaic:
        surcharges["mosaic"] = 0.0
    if args.no_fliplr:
        surcharges["fliplr"] = 0.0

    print(f"modèle {args.model} · {args.epochs} epochs · imgsz {args.imgsz} · batch {args.batch}")
    if surcharges:
        print(f"augmentations désactivées : {', '.join(surcharges)}")

    modele = YOLO(args.model)
    modele.train(
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
        **surcharges,
    )

    poids = args.project / args.name / "weights" / "best.pt"
    print(f"\nPoids : {poids}")
    print(f"Évaluation détaillée :\n  uv run python detection/src/evaluate.py --weights {poids}")


if __name__ == "__main__":
    main()
