#!/usr/bin/env python3
"""Construit un dataset YOLO dérivé, propre, à partir de la livraison BnF.

La livraison (`20250214-segmentation-dataset-iiif/`) est traitée en lecture seule :
rien n'y est écrit. Le dérivé est produit dans `data/`, avec des liens symboliques
vers les images pour ne pas dupliquer 1,1 Go.

Nettoyages appliqués :
  - filtrage des classes (par défaut : Illustration seule, cf. `nc=1`)
  - suppression des doublons exacts (46 relevés dans la livraison)
  - suppression des boîtes d'aire nulle (1 relevée)
  - les fichiers label vides sont conservés : ce sont des négatifs purs

Usage :
    uv run python src/prepare_dataset.py                # nc=1, illustrations seules
    uv run python src/prepare_dataset.py --classes 0 1  # nc=2, pour l'ablation
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DATA, LIVRAISON

SOURCE = LIVRAISON
NOMS_SOURCE = {0: "Illustration", 1: "Texte"}
SPLITS = ("train", "val")


def lire_boites(chemin: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse un fichier label YOLO. Les lignes malformées sont signalées, pas ignorées."""
    boites = []
    for num, ligne in enumerate(chemin.read_text().splitlines(), 1):
        if not ligne.strip():
            continue
        champs = ligne.split()
        if len(champs) != 5:
            raise ValueError(f"{chemin.name} l.{num} : {len(champs)} champs au lieu de 5")
        cls, *coords = champs
        boites.append((int(cls), *map(float, coords)))
    return boites


def nettoyer(boites, classes_gardees):
    """Filtre les classes, déduplique, écarte les boîtes dégénérées.

    Renvoie (boites_propres, compteurs) pour pouvoir rendre compte du nettoyage.
    """
    gardees, vues = [], set()
    n_filtrees = n_doublons = n_degenerees = 0

    for cls, cx, cy, w, h in boites:
        if cls not in classes_gardees:
            n_filtrees += 1
            continue
        if w <= 0 or h <= 0:
            n_degenerees += 1
            continue
        cle = (cls, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
        if cle in vues:
            n_doublons += 1
            continue
        vues.add(cle)
        # réindexation : les classes gardées deviennent 0..n-1, dans l'ordre d'origine
        gardees.append((classes_gardees.index(cls), cx, cy, w, h))

    return gardees, (n_filtrees, n_doublons, n_degenerees)


def preparer(sortie: Path, classes: list[int], forcer: bool) -> None:
    if sortie.exists():
        if not forcer:
            raise SystemExit(f"{sortie} existe déjà — relancer avec --force pour l'écraser.")
        shutil.rmtree(sortie)

    noms = [NOMS_SOURCE[c] for c in classes]
    print(f"Classes conservées : {noms}  (nc={len(classes)})")
    print(f"Source  : {SOURCE}")
    print(f"Sortie  : {sortie}\n")

    total = {"boites": 0, "filtrees": 0, "doublons": 0, "degenerees": 0}

    for split in SPLITS:
        dst_img = sortie / split / "images"
        dst_lbl = sortie / split / "labels"
        dst_img.mkdir(parents=True)
        dst_lbl.mkdir(parents=True)

        src_img = SOURCE / split / "images"
        src_lbl = SOURCE / split / "labels"

        n_vues = n_vides = n_boites = 0
        for fichier in sorted(src_lbl.glob("*.txt")):
            propres, (f_, d_, g_) = nettoyer(lire_boites(fichier), classes)
            total["filtrees"] += f_
            total["doublons"] += d_
            total["degenerees"] += g_

            # un fichier vide reste un fichier : c'est un négatif pur, pas un label manquant
            (dst_lbl / fichier.name).write_text(
                "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in propres)
            )

            image = src_img / f"{fichier.stem}.jpg"
            if not image.exists():
                raise SystemExit(f"image manquante pour {fichier.name}")
            (dst_img / image.name).symlink_to(image.resolve())

            n_vues += 1
            n_boites += len(propres)
            n_vides += not propres

        total["boites"] += n_boites
        print(f"  {split:<6} {n_vues:>5} vues · {n_boites:>6} boîtes · {n_vides:>4} vues sans boîte (négatifs)")

    yaml = sortie / "dataset.yaml"
    yaml.write_text(
        "# Généré par prepare_dataset.py — chemins absolus, machine locale\n"
        f"path: {sortie.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(classes)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(noms))
    )

    print(f"\nNettoyage : {total['filtrees']} boîtes filtrées (classe écartée) · "
          f"{total['doublons']} doublons · {total['degenerees']} dégénérées")
    print(f"Total conservé : {total['boites']} boîtes")
    print(f"\nConfig écrite : {yaml}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", type=int, nargs="+", default=[0],
                   help="classes source à conserver (0=Illustration, 1=Texte). Défaut : 0")
    p.add_argument("--out", type=Path, default=None,
                   help="dossier de sortie. Défaut : data/detection-nc<N>")
    p.add_argument("--force", action="store_true", help="écrase le dossier de sortie s'il existe")
    args = p.parse_args()

    inconnues = set(args.classes) - set(NOMS_SOURCE)
    if inconnues:
        raise SystemExit(f"classes inconnues : {sorted(inconnues)}")

    sortie = args.out or DATA / f"detection-nc{len(args.classes)}"
    preparer(sortie, args.classes, args.force)


if __name__ == "__main__":
    main()
