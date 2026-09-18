# Illustration detection — evaluation and model

We detect **illustrations**: zero, one or several per view. The boxes produced then feed
the classification stage, which is out of scope for this document.

Last updated: 2026-09-17

---

## 1. Evaluation metrics

This is the part that decides. As long as these metrics are not implemented and frozen,
launching a training run is pointless: the results would not be comparable.

### IoU, the foundation of all metrics

The **IoU** (intersection over union) between a predicted box and a ground-truth box is
the **matching criterion** of the whole chain:

- **AP50** = average precision, counting as correct any prediction with **IoU ≥ 0.50**
- **mAP50-95** = the same, averaged over ten IoU thresholds from 0.50 to 0.95

That is: **IoU → matching → TP/FP/FN counts → precision/recall → AP**.

**Why not report it alone.** It can only be computed between two already-matched boxes,
so it is blind to missed illustrations and to invented ones.
Example: a page contains 3, the model predicts only one with IoU 0.95.
Mean IoU 0.95 — excellent; real recall 33%.

### Primary metric

**Precision at fixed recall** (typically P@R=0.90).

The business problem is **illustration hallucination**: the dataset ships 372 views where
the previous model invented one. We therefore want to know how many false detections a
given recall level costs.

### Standard metrics

| Metric | Why |
|---|---|
| **mAP50-95** | Comparability with the literature |
| **AP50** | More direct reading of detection quality |
| **Precision / recall / F1** at a fixed threshold | The real operating point |

### Stratifications — mandatory

**By box size.** The area distribution is bimodal: p25 = 3% of the page,
p75 = 74%. Large boxes are easy and dominate the average — **without stratification,
the "small illustrations" regime is invisible.**

**By content tag** (`photographie`, `comic_book`, `film_roll`, `plan`…): shows which
sub-population the model fails on.

### Problem-specific metrics

| Metric | What it captures |
|---|---|
| **False-positive rate on views without illustrations** | 398 views contain none. Direct, readable measure of hallucination |
| **Mean IoU of matched boxes** | A box that is too loose yields a crop unusable for downstream classification |

### Measurement conditions

- **Split the test set by `parent_ark`, never by view.** Several views of the same
  work are nearly identical: splitting by view guarantees leakage.
- **Labels cleaned upstream**: 22 exact duplicates and 1 zero-area box.
  Checked on 2026-09-17, no other anomaly across the 11,051 `Illustration` boxes.
- **Keep views without illustrations**: their empty label makes them pure negatives,
  and the basis of the false-positive metric above.

---

## 2. Model

**YOLO26l.** Current Ultralytics generation (January 2026). Two of its contributions
target our difficulties directly: **STAL**, a label assignment designed for small
targets — we have ~1,100 under 1% of the page — and natively **NMS-free** inference.

**NMS ablation, at no cost.** The model is evaluated with `nms=False` **and** `nms=True`
on the same trained weights. Answers a real question: does NMS create duplicates on
full-page illustrations?

AGPL-3.0 license, validated for this project.

---

## 3. Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `imgsz` | 800 | Corpus resolution, and a multiple of 32. **239 views (5%) are not 800×800** — mostly `-f1` covers: they will be letterboxed |
| `epochs` | 50 | First training run |
| `batch` | 6 | Measured: 5.1 GB of VRAM at `batch=4` out of the 8.3 GB available. 6 keeps a margin |
| `seed` | fixed | Reproducibility |
| `nc` | 1 | Illustrations only |

**One class or two?** The provided labels also contain a `Texte` class (70% of the
boxes), which we do not deliver. We train with `nc=1`: text blocks become background,
and the model learns "do not detect here". An `nc=2` run remains to be measured as an
ablation — with a known handicap: **`Texte` is only annotated on 83% of views**, so
training on it would penalize the model on text that nobody outlined.

**Augmentations — to review before launching.** Ultralytics defaults target natural
photos: `fliplr` mirrors the page text, `mosaic` fabricates layouts that do not exist in
the corpus.
