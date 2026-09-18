#!/usr/bin/env python3
"""Évaluation détaillée d'un détecteur d'illustrations.

Calcule les métriques définies dans docs/modeles-detection.md, qu'Ultralytics ne
fournit pas : précision à rappel fixé, stratification par taille de boîte et par tag
de contenu, taux de faux positifs sur les vues sans illustration, IoU moyenne des
boîtes appariées.

Usage :
    uv run python detection/src/evaluate.py --weights detection/runs/yolo26l/weights/best.pt
    uv run python detection/src/evaluate.py --weights ... --conf 0.4 --json resultats.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import DATA, LIVRAISON, configurer_ultralytics

# Strates d'aire, en fraction de page. Bornes lisibles plutôt que quantiles bruts :
# la distribution est bimodale, ces coupures séparent les régimes réels du corpus.
STRATES = [
    ("minuscule  (<1%)", 0.0, 0.01),
    ("petite  (1-10%)", 0.01, 0.10),
    ("moyenne (10-50%)", 0.10, 0.50),
    ("grande    (>50%)", 0.50, 1.01),
]

# Tags de workflow, à ne pas confondre avec des tags de contenu
TAGS_WORKFLOW = {"To check", "to check", "To delete", "to enjoy", "doublon", "last"}


# ---------------------------------------------------------------- chargement

def charger_gt(dossier_labels: Path) -> dict[str, np.ndarray]:
    """Lit les labels YOLO et renvoie {nom_vue: boîtes xyxy normalisées}."""
    gt = {}
    for fichier in sorted(dossier_labels.glob("*.txt")):
        boites = []
        for ligne in fichier.read_text().splitlines():
            if not ligne.strip():
                continue
            _, cx, cy, w, h = map(float, ligne.split())
            boites.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        gt[fichier.stem] = np.array(boites, dtype=np.float64).reshape(-1, 4)
    return gt


def charger_tags() -> dict[str, list[str]]:
    """Tags de contenu par vue, depuis la vérité terrain (pas depuis items.jsonlines)."""
    chemin = LIVRAISON / "annotations.jsonlines"
    if not chemin.exists():
        return {}
    tags = {}
    with chemin.open() as f:  # 23 Mo : lecture en streaming, ligne par ligne
        for ligne in f:
            d = json.loads(ligne)
            uuid = d.get("item", {}).get("uuid") or d.get("uuid")
            if uuid:
                tags[uuid] = [t for t in (d.get("tags") or []) if t not in TAGS_WORKFLOW]
    return tags


# ---------------------------------------------------------------- géométrie

def ious(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Matrice IoU (n_pred, n_gt) sur des boîtes xyxy."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aire(pred)[:, None] + aire(gt)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def apparier(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, seuil_iou: float):
    """Appariement glouton par score décroissant, une prédiction par boîte vraie.

    Renvoie, pour chaque prédiction triée : (est_tp, indice_gt_apparie ou -1, iou).
    """
    ordre = np.argsort(-scores)
    m = ious(pred[ordre], gt) if len(pred) else np.zeros((0, len(gt)))
    pris = np.zeros(len(gt), dtype=bool)
    resultats = []
    for i in range(len(ordre)):
        j, meilleur = -1, seuil_iou
        for k in range(len(gt)):
            if not pris[k] and m[i, k] >= meilleur:
                j, meilleur = k, m[i, k]
        if j >= 0:
            pris[j] = True
        resultats.append((j >= 0, j, meilleur if j >= 0 else 0.0))
    return ordre, resultats


# ---------------------------------------------------------------- métriques

def courbe_pr(tp: np.ndarray, scores: np.ndarray, n_gt: int):
    """Courbe précision-rappel, triée par score décroissant."""
    if n_gt == 0 or len(tp) == 0:
        return np.array([]), np.array([]), np.array([])
    ordre = np.argsort(-scores)
    tp = tp[ordre]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    rappel = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return precision, rappel, scores[ordre]


def ap_101(precision: np.ndarray, rappel: np.ndarray) -> float:
    """Average precision, interpolation COCO à 101 points."""
    if len(precision) == 0:
        return 0.0
    # enveloppe décroissante : la précision ne peut pas remonter vers la droite
    p = np.concatenate([[0.0], precision, [0.0]])
    r = np.concatenate([[0.0], rappel, [rappel[-1]]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    points = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(points, r, p, left=p[0], right=0.0)))


def precision_a_rappel(precision: np.ndarray, rappel: np.ndarray, cible: float):
    """Précision atteignable au rappel demandé. None si le rappel cible est hors de portée."""
    if len(rappel) == 0 or rappel[-1] < cible:
        return None
    p = precision.copy()
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    return float(p[np.searchsorted(rappel, cible, side="left")])


def evaluer(enregs: list[dict], n_gt: int, seuils_iou=(0.5,)) -> dict:
    """Agrège les appariements en métriques. `enregs` = une entrée par prédiction."""
    if not enregs:
        return {"n_gt": n_gt, "ap50": 0.0, "map50_95": 0.0, "p_at_r90": None}
    scores = np.array([e["score"] for e in enregs])
    res = {"n_gt": n_gt, "n_pred": len(enregs)}

    tp50 = np.array([e["iou"] >= 0.5 for e in enregs])
    p, r, _ = courbe_pr(tp50, scores, n_gt)
    res["ap50"] = ap_101(p, r)
    res["p_at_r90"] = precision_a_rappel(p, r, 0.90)
    res["rappel_max"] = float(r[-1]) if len(r) else 0.0

    aps = []
    for s in np.arange(0.5, 1.0, 0.05):
        tp = np.array([e["iou"] >= s for e in enregs])
        pp, rr, _ = courbe_pr(tp, scores, n_gt)
        aps.append(ap_101(pp, rr))
    res["map50_95"] = float(np.mean(aps))
    return res


# ---------------------------------------------------------------- rapport

def ligne(nom: str, m: dict, largeur: int = 20) -> str:
    par = f"{m['p_at_r90']:.3f}" if m.get("p_at_r90") is not None else f"— (max {m.get('rappel_max',0):.2f})"
    return (f"  {nom:<{largeur}} n={m['n_gt']:>5}  AP50={m['ap50']:.3f}  "
            f"mAP50-95={m['map50_95']:.3f}  P@R90={par}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA / "detection-nc1")
    p.add_argument("--split", default="val")
    p.add_argument("--conf", type=float, default=0.25, help="seuil du point de fonctionnement")
    p.add_argument("--iou-nms", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--nms", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-nms pour l'inférence end-to-end de YOLO26")
    p.add_argument("--json", type=Path, default=None, help="écrit les résultats au format JSON")
    args = p.parse_args()

    images = sorted((args.data / args.split / "images").glob("*.jpg"))
    gt = charger_gt(args.data / args.split / "labels")
    tags = charger_tags()
    print(f"{len(images)} vues · {sum(len(b) for b in gt.values())} boîtes de vérité terrain")
    print(f"seuil de fonctionnement : conf={args.conf}  ·  NMS : {'oui' if args.nms else 'non (end-to-end)'}\n")

    configurer_ultralytics()
    from ultralytics import YOLO
    modele = YOLO(args.weights)

    enregs: list[dict] = []          # une entrée par prédiction, tous seuils confondus
    n_gt_total = 0
    par_strate: dict[str, list] = defaultdict(list)
    n_gt_strate: dict[str, int] = defaultdict(int)
    par_tag: dict[str, list] = defaultdict(list)
    n_gt_tag: dict[str, int] = defaultdict(int)
    vues_vides = fp_vues_vides = 0
    ious_apparies: list[float] = []

    # conf très bas : il faut toute la courbe PR, pas seulement le point de fonctionnement
    for lot in range(0, len(images), 16):
        chemins = images[lot:lot + 16]
        sorties = modele.predict(chemins, imgsz=args.imgsz, conf=0.001, iou=args.iou_nms,
                                 nms=args.nms, max_det=300, verbose=False)
        for chemin, sortie in zip(chemins, sorties):
            nom = chemin.stem
            boites_gt = gt[nom]
            n_gt_total += len(boites_gt)
            h, w = sortie.orig_shape
            xyxy = sortie.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h])
            scores = sortie.boxes.conf.cpu().numpy()

            aires_gt = (boites_gt[:, 2] - boites_gt[:, 0]) * (boites_gt[:, 3] - boites_gt[:, 1]) if len(boites_gt) else np.array([])
            etiq_gt = [next(n for n, lo, hi in STRATES if lo <= a < hi) for a in aires_gt]
            for e in etiq_gt:
                n_gt_strate[e] += 1
            tags_vue = tags.get(nom, [])
            for t in tags_vue:
                n_gt_tag[t] += len(boites_gt)

            # vues sans illustration : base du taux de faux positifs
            if len(boites_gt) == 0:
                vues_vides += 1
                if (scores >= args.conf).any():
                    fp_vues_vides += 1

            ordre, apps = apparier(xyxy, scores, boites_gt, seuil_iou=0.5)
            for rang, (est_tp, idx_gt, iou) in enumerate(apps):
                s = float(scores[ordre[rang]])
                b = xyxy[ordre[rang]]
                aire_pred = (b[2] - b[0]) * (b[3] - b[1])
                enr = {"score": s, "iou": float(iou)}
                enregs.append(enr)
                if est_tp and s >= args.conf:
                    ious_apparies.append(float(iou))
                # strate : celle de la boîte vraie si appariée, sinon celle de la prédiction
                aire_ref = aires_gt[idx_gt] if est_tp else aire_pred
                strate = next(n for n, lo, hi in STRATES if lo <= aire_ref < hi)
                par_strate[strate].append(enr)
                for t in tags_vue:
                    par_tag[t].append(enr)

    print("=" * 78)
    print("GLOBAL")
    glob = evaluer(enregs, n_gt_total)
    print(ligne("toutes tailles", glob))

    # point de fonctionnement
    au_seuil = [e for e in enregs if e["score"] >= args.conf]
    tp = sum(1 for e in au_seuil if e["iou"] >= 0.5)
    fp = len(au_seuil) - tp
    prec = tp / max(len(au_seuil), 1)
    rapp = tp / max(n_gt_total, 1)
    f1 = 2 * prec * rapp / max(prec + rapp, 1e-12)
    print(f"\n  au seuil conf={args.conf} :  précision {prec:.3f} · rappel {rapp:.3f} · F1 {f1:.3f}"
          f"   (TP {tp} · FP {fp})")
    print(f"  IoU moyenne des boîtes appariées : {np.mean(ious_apparies):.3f}" if ious_apparies else "  aucune boîte appariée")
    if vues_vides:
        print(f"  faux positifs sur vues SANS illustration : {fp_vues_vides}/{vues_vides} "
              f"({fp_vues_vides/vues_vides:.1%})")

    print("\n" + "=" * 78)
    print("PAR TAILLE DE BOÎTE")
    strates_res = {}
    for nom, _, _ in STRATES:
        m = evaluer(par_strate[nom], n_gt_strate[nom])
        strates_res[nom] = m
        print(ligne(nom, m))

    print("\n" + "=" * 78)
    print("PAR TAG DE CONTENU  (10 plus fréquents)")
    tags_res = {}
    frequents = sorted(n_gt_tag.items(), key=lambda x: -x[1])[:10]
    for t, _ in frequents:
        m = evaluer(par_tag[t], n_gt_tag[t])
        tags_res[t] = m
        print(ligne(t, m, largeur=34))

    if args.json:
        args.json.write_text(json.dumps({
            "poids": str(args.weights), "split": args.split, "conf": args.conf, "nms": args.nms,
            "global": glob,
            "point_de_fonctionnement": {"precision": prec, "rappel": rapp, "f1": f1,
                                        "tp": tp, "fp": fp,
                                        "iou_moyenne": float(np.mean(ious_apparies)) if ious_apparies else None,
                                        "fp_vues_vides": fp_vues_vides, "vues_vides": vues_vides},
            "par_taille": strates_res, "par_tag": tags_res,
        }, indent=2, ensure_ascii=False))
        print(f"\nRésultats écrits dans {args.json}")


if __name__ == "__main__":
    main()
