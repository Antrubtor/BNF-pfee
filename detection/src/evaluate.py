#!/usr/bin/env python3
"""Detailed evaluation of an illustration detector.

Computes the metrics defined in docs/modeles-detection.md that Ultralytics does not
provide: precision at fixed recall, stratification by box size and by content tag,
false-positive rate on views without illustrations, mean IoU of matched boxes.

Usage:
    uv run python detection/src/evaluate.py --weights detection/runs/yolo26l/weights/best.pt
    uv run python detection/src/evaluate.py --weights ... --conf 0.4 --json results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import DATA, DELIVERY, configure_ultralytics

# Area strata, as a fraction of the page. Readable bounds rather than raw quantiles:
# the distribution is bimodal, these cuts separate the real regimes of the corpus.
STRATA = [
    ("tiny      (<1%)", 0.0, 0.01),
    ("small    (1-10%)", 0.01, 0.10),
    ("medium  (10-50%)", 0.10, 0.50),
    ("large     (>50%)", 0.50, 1.01),
]

# Workflow tags, not to be confused with content tags
WORKFLOW_TAGS = {"To check", "to check", "To delete", "to enjoy", "doublon", "last"}


# ---------------------------------------------------------------- loading

def load_gt(labels_dir: Path) -> dict[str, np.ndarray]:
    """Read YOLO labels and return {view_name: normalized xyxy boxes}."""
    gt = {}
    for file in sorted(labels_dir.glob("*.txt")):
        boxes = []
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = map(float, line.split())
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        gt[file.stem] = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    return gt


def load_tags() -> dict[str, list[str]]:
    """Content tags per view, from the ground truth (not from items.jsonlines)."""
    path = DELIVERY / "annotations.jsonlines"
    if not path.exists():
        return {}
    tags = {}
    with path.open() as f:  # 23 MB: stream line by line
        for line in f:
            d = json.loads(line)
            uuid = d.get("item", {}).get("uuid") or d.get("uuid")
            if uuid:
                tags[uuid] = [t for t in (d.get("tags") or []) if t not in WORKFLOW_TAGS]
    return tags


# ---------------------------------------------------------------- geometry

def ious(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU matrix (n_pred, n_gt) over xyxy boxes."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area(pred)[:, None] + area(gt)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float):
    """Greedy matching by decreasing score, one prediction per ground-truth box.

    Returns, for each sorted prediction: (is_tp, matched_gt_index or -1, iou).
    """
    order = np.argsort(-scores)
    m = ious(pred[order], gt) if len(pred) else np.zeros((0, len(gt)))
    taken = np.zeros(len(gt), dtype=bool)
    results = []
    for i in range(len(order)):
        j, best = -1, iou_thr
        for k in range(len(gt)):
            if not taken[k] and m[i, k] >= best:
                j, best = k, m[i, k]
        if j >= 0:
            taken[j] = True
        results.append((j >= 0, j, best if j >= 0 else 0.0))
    return order, results


# ---------------------------------------------------------------- metrics

def pr_curve(tp: np.ndarray, scores: np.ndarray, n_gt: int):
    """Precision-recall curve, sorted by decreasing score."""
    if n_gt == 0 or len(tp) == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(-scores)
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return precision, recall, scores[order]


def ap_101(precision: np.ndarray, recall: np.ndarray) -> float:
    """Average precision, COCO 101-point interpolation."""
    if len(precision) == 0:
        return 0.0
    # decreasing envelope: precision cannot rise going right
    p = np.concatenate([[0.0], precision, [0.0]])
    r = np.concatenate([[0.0], recall, [recall[-1]]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    points = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(points, r, p, left=p[0], right=0.0)))


def precision_at_recall(precision: np.ndarray, recall: np.ndarray, target: float):
    """Precision reachable at the requested recall. None if the target recall is out of reach."""
    if len(recall) == 0 or recall[-1] < target:
        return None
    p = precision.copy()
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    return float(p[np.searchsorted(recall, target, side="left")])


def evaluate(records: list[dict], n_gt: int) -> dict:
    """Aggregate matches into metrics. `records` = one entry per prediction."""
    if not records:
        return {"n_gt": n_gt, "ap50": 0.0, "map50_95": 0.0, "p_at_r90": None}
    scores = np.array([e["score"] for e in records])
    res = {"n_gt": n_gt, "n_pred": len(records)}

    tp50 = np.array([e["iou"] >= 0.5 for e in records])
    p, r, _ = pr_curve(tp50, scores, n_gt)
    res["ap50"] = ap_101(p, r)
    res["p_at_r90"] = precision_at_recall(p, r, 0.90)
    res["max_recall"] = float(r[-1]) if len(r) else 0.0

    aps = []
    for s in np.arange(0.5, 1.0, 0.05):
        tp = np.array([e["iou"] >= s for e in records])
        pp, rr, _ = pr_curve(tp, scores, n_gt)
        aps.append(ap_101(pp, rr))
    res["map50_95"] = float(np.mean(aps))
    return res


# ---------------------------------------------------------------- report

def row(name: str, m: dict, width: int = 20) -> str:
    par = f"{m['p_at_r90']:.3f}" if m.get("p_at_r90") is not None else f"— (max {m.get('max_recall',0):.2f})"
    return (f"  {name:<{width}} n={m['n_gt']:>5}  AP50={m['ap50']:.3f}  "
            f"mAP50-95={m['map50_95']:.3f}  P@R90={par}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA / "detection-nc1")
    p.add_argument("--split", default="val")
    p.add_argument("--conf", type=float, default=0.25, help="operating-point threshold")
    p.add_argument("--iou-nms", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--nms", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-nms for YOLO26 end-to-end inference")
    p.add_argument("--json", type=Path, default=None, help="write results as JSON")
    args = p.parse_args()

    images = sorted((args.data / args.split / "images").glob("*.jpg"))
    gt = load_gt(args.data / args.split / "labels")
    tags = load_tags()
    print(f"{len(images)} views · {sum(len(b) for b in gt.values())} ground-truth boxes")
    print(f"operating point: conf={args.conf}  ·  NMS: {'yes' if args.nms else 'no (end-to-end)'}\n")

    configure_ultralytics()
    from ultralytics import YOLO
    model = YOLO(args.weights)

    records: list[dict] = []          # one entry per prediction, all thresholds pooled
    n_gt_total = 0
    by_stratum: dict[str, list] = defaultdict(list)
    n_gt_stratum: dict[str, int] = defaultdict(int)
    by_tag: dict[str, list] = defaultdict(list)
    n_gt_tag: dict[str, int] = defaultdict(int)
    empty_views = fp_empty_views = 0
    matched_ious: list[float] = []

    # very low conf: the whole PR curve is needed, not just the operating point
    for start in range(0, len(images), 16):
        paths = images[start:start + 16]
        outputs = model.predict(paths, imgsz=args.imgsz, conf=0.001, iou=args.iou_nms,
                                nms=args.nms, max_det=300, verbose=False)
        for path, output in zip(paths, outputs):
            name = path.stem
            gt_boxes = gt[name]
            n_gt_total += len(gt_boxes)
            h, w = output.orig_shape
            xyxy = output.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h])
            scores = output.boxes.conf.cpu().numpy()

            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) if len(gt_boxes) else np.array([])
            gt_labels = [next(n for n, lo, hi in STRATA if lo <= a < hi) for a in gt_areas]
            for e in gt_labels:
                n_gt_stratum[e] += 1
            view_tags = tags.get(name, [])
            for t in view_tags:
                n_gt_tag[t] += len(gt_boxes)

            # views without illustrations: basis of the false-positive rate
            if len(gt_boxes) == 0:
                empty_views += 1
                if (scores >= args.conf).any():
                    fp_empty_views += 1

            order, matches = match(xyxy, scores, gt_boxes, iou_thr=0.5)
            for rank, (is_tp, gt_idx, iou) in enumerate(matches):
                s = float(scores[order[rank]])
                b = xyxy[order[rank]]
                pred_area = (b[2] - b[0]) * (b[3] - b[1])
                rec = {"score": s, "iou": float(iou)}
                records.append(rec)
                if is_tp and s >= args.conf:
                    matched_ious.append(float(iou))
                # stratum: the matched ground-truth box's if matched, else the prediction's
                ref_area = gt_areas[gt_idx] if is_tp else pred_area
                stratum = next(n for n, lo, hi in STRATA if lo <= ref_area < hi)
                by_stratum[stratum].append(rec)
                for t in view_tags:
                    by_tag[t].append(rec)

    print("=" * 78)
    print("GLOBAL")
    overall = evaluate(records, n_gt_total)
    print(row("all sizes", overall))

    # operating point
    at_thr = [e for e in records if e["score"] >= args.conf]
    tp = sum(1 for e in at_thr if e["iou"] >= 0.5)
    fp = len(at_thr) - tp
    prec = tp / max(len(at_thr), 1)
    rec_ = tp / max(n_gt_total, 1)
    f1 = 2 * prec * rec_ / max(prec + rec_, 1e-12)
    print(f"\n  at conf={args.conf}:  precision {prec:.3f} · recall {rec_:.3f} · F1 {f1:.3f}"
          f"   (TP {tp} · FP {fp})")
    print(f"  mean IoU of matched boxes: {np.mean(matched_ious):.3f}" if matched_ious else "  no matched box")
    if empty_views:
        print(f"  false positives on views WITHOUT illustrations: {fp_empty_views}/{empty_views} "
              f"({fp_empty_views/empty_views:.1%})")

    print("\n" + "=" * 78)
    print("BY BOX SIZE")
    strata_res = {}
    for name, _, _ in STRATA:
        m = evaluate(by_stratum[name], n_gt_stratum[name])
        strata_res[name] = m
        print(row(name, m))

    print("\n" + "=" * 78)
    print("BY CONTENT TAG  (10 most frequent)")
    tags_res = {}
    frequent = sorted(n_gt_tag.items(), key=lambda x: -x[1])[:10]
    for t, _ in frequent:
        m = evaluate(by_tag[t], n_gt_tag[t])
        tags_res[t] = m
        print(row(t, m, width=34))

    if args.json:
        args.json.write_text(json.dumps({
            "weights": str(args.weights), "split": args.split, "conf": args.conf, "nms": args.nms,
            "global": overall,
            "operating_point": {"precision": prec, "recall": rec_, "f1": f1,
                                "tp": tp, "fp": fp,
                                "mean_iou": float(np.mean(matched_ious)) if matched_ious else None,
                                "fp_empty_views": fp_empty_views, "empty_views": empty_views},
            "by_size": strata_res, "by_tag": tags_res,
        }, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
