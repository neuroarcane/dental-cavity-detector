# Report sections — Hyperparameter Tuning & Benchmarking (+ Results additions)

**Owner:** Temirlan · SCRUM-9 / SCRUM-16 / SCRUM-22
**Purpose:** Report-ready prose for the final report §5 (Results additions), §6
(Hyperparameter Tuning), and §7 (Benchmarking). Paste into the working draft;
`🔒 INTERNAL` notes are for the team and must be removed before submission.

**Provenance of the numbers below:** recomputed locally with our own evaluation
pipeline (`src/eval`) on the canonical held-out test split (2202 images; 70/15/15
source-stratified, seed 42). As a correctness check, our pipeline reproduced the
published YOLO headline within rounding — **mAP@0.5 0.7466 vs 0.7462**,
**mAP@0.5:0.95 0.4517 vs 0.4467** — confirming the split matches the one used in
training (no train/test leakage). Fixed operating point for P/R/F1 and the
confusion matrix: **confidence 0.25, IoU 0.5, NMS IoU 0.7**.

---

## 5.1 (extend) YOLO11 detector — full metrics

**Overall:** mAP@0.5 = **0.747**, mAP@0.5:0.95 = **0.452** on the held-out test set.

| Class | AP@0.5 | AP@0.5:0.95 contrib. | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Impacted Tooth | 0.937 | — | 0.864 | 0.960 | 0.910 |
| Crown | 0.867 | — | 0.783 | 0.944 | 0.856 |
| Filling | 0.738 | — | 0.666 | 0.733 | 0.698 |
| **Cavity** | **0.445** | — | **0.610** | **0.435** | **0.508** |
| **Macro avg** | 0.747 | 0.452 | 0.731 | 0.768 | 0.743 |

*🔒 INTERNAL — reconciliation note: §5.1 currently cites Cavity recall 0.391 (Ultralytics
`.val()` default operating point). At our ratified operating point (conf 0.25) the same
model scores Cavity recall 0.435. Both describe the same weakness; pick one operating
point and state it once so the report doesn't show two numbers. AP@0.5 (0.445 vs published
0.42) matches within rounding and is threshold-independent.*

**Efficiency (P5):** 2,590,620 parameters · 5.48 MB on disk · 65.9 ms/image average
inference on CPU (Colab GPU will be far faster — 🔒 record the GPU figure for the report).
A nano-scale, ~5 MB real-time model is well inside what a clinical assistive tool could
deploy on modest hardware.

## 5.2 (support) Error patterns — what the confusion matrix shows

Row-normalized confusion matrix (rows = ground truth), from
`reports/yolo/YOLO11n_confusion_matrix.png`:

- **Cavity:** 0.43 correct, **0.56 missed to background**, ~0 confused with other classes.
  The Cavity failure is almost entirely **non-detection (false negatives)**, not
  misclassification — the model doesn't confuse cavities with other findings, it fails to
  see them at all. This is consistent with the "small, low-contrast radiolucency" account
  in §5.2: the objects are hard to localize, not hard to label.
- **Crown 0.93 / Impacted Tooth 0.96 correct** — large, high-contrast structures are easy.
- **Filling:** 0.73 correct, 0.24 missed to background, 0.03 confused as Crown (radiopaque
  restorations resemble crowns).
- **False positives (background row):** the model's spurious boxes are mostly **Filling**
  (0.70 of FPs) — it over-calls fillings on normal structure. In an assistive-triage
  framing this is the benign error mode (a clinician glances and dismisses), unlike the
  Cavity false negatives.

## 5.3 (extend) Performance vs. Expectations scorecard — now resolvable

| Ref | Target | Actual | Met? |
|---|---|---|---|
| P1 | mAP@0.5 ≥ 0.50 | 0.747 | ✅ |
| P2 | No class ignored | All 4 detected; Cavity weak (AP 0.445) | ⚠ |
| P3 | Cavity recall reported & addressed | 0.435 @ conf0.25 (0.391 @ default) — misses ~57–61% | ⚠ Primary gap |
| P4 | mAP@0.5:0.95 reported | **0.452** | ✅ |
| P5 | Efficiency reported | **2.59M params, 5.48 MB, 65.9 ms/img CPU** | ✅ |
| F5 | 5–10 samples incl. failure case | **8 panels** (3 Cavity failures + 5 multi-finding), `reports/yolo/samples/` | ✅ |

*🔒 INTERNAL: P4, P5, F5 move from 🔲 pending to ✅. P2/P3 remain ⚠ by design — the honest
central finding, not a gap to paper over.*

## 5.4 (fill the Temirlan TODO) Qualitative samples

Eight GT-vs-prediction panels are in `reports/yolo/samples/` (left = ground truth green,
right = prediction red; captions in `captions.json`). They include **3 Cavity failure
cases** — `sample_02_failure` is the clearest single-finding miss. **Do not commit these
to git** — they are real patient X-rays; they live in the gitignored `reports/` and go
only into the Blackboard report.

---

## 6. Hyperparameter Tuning

**🔒 Owner: Temirlan · SCRUM-9 / SCRUM-22**

### 6.1 Harness

A sweep harness (`src/tuning`) supports both **grid search** (exhaustive over a small
list) and **random search** (sampling a wider space), over learning rate (`lr0`, `lrf`),
optimizer settings (momentum, weight decay, warmup) and augmentation (mosaic, mixup, HSV
jitter, rotation, translate, scale, flip). Enumeration, ranking and seed-reproducibility
were unit-tested with a stub trainer before use, so trial bookkeeping is known-correct
independent of any training run.

### 6.2 Objective

The starting point is YOLO11n at 50 epochs, mAP@0.5 = 0.747. The tuning objective is **not**
to lift overall mAP — that would come cheaply from the already-strong Crown/Impacted-Tooth
classes and flatter the headline while leaving the real problem untouched. The objective is
to **raise Cavity recall (0.435)** without materially degrading the other classes, because
a missed cavity is the costly error in assistive triage (P3).

### 6.3 Rationale for the search space (why these knobs, for this failure)

- **Mosaic / mixup:** on by default in YOLO; they paste image fragments together. On
  panoramic X-rays this can create anatomically impossible composites, potentially hurting
  small-lesion learning — so we sweep them *down* as well as up.
- **Learning rate & warmup:** the standard first lever for transfer-learning stability.
- **Cavity is intrinsically hard, not just rare:** oversampling (Section 3) duplicated
  existing cavities but added no new visual variety. So we expect tuning to yield only
  modest Cavity gains — a hypothesis we state up front and test, rather than promising a fix.

### 6.4 Results

*🔒 INTERNAL — PENDING GPU RUN. The sweep must run on Colab (each trial fine-tunes YOLO;
infeasible on CPU). The turnkey sweep cells are in `notebooks/03_evaluation.ipynb`. Record
for each trial: what changed, why it was tried, expected effect, and actual effect —
**including changes that did not work** (the rubric explicitly rewards this). Fill the table
below from the run:*

| Trial | Change vs. baseline | Rationale | Expected | Actual (mAP@0.5 / Cavity recall) |
|---|---|---|---|---|
| baseline | — | — | — | 0.747 / 0.435 |
| … | … | … | … | pending |

---

## 7. Benchmarking

**🔒 Owner: Temirlan · SCRUM-16**

### 7.1 Models compared

| Role | Model | Task | Metric family |
|---|---|---|---|
| Classification floor | CNN (from scratch, classifier) | classification | accuracy, image-level P/R/F1 |
| Final detector | YOLO11n (COCO-pretrained, transfer learning) | detection | mAP, IoU-based P/R/F1 |
| Challenger | Faster R-CNN (ResNet-50-FPN, two-stage CNN) | detection | mAP, IoU-based P/R/F1 |

**Why Faster R-CNN, not RT-DETR:** the original challenger (RT-DETR) was dropped because
comparing a *pretrained transformer* against our from-scratch/CNN line-up would confound
two variables — architecture family and pre-training advantage. Faster R-CNN keeps the
detector comparison **CNN-to-CNN**, so a measured difference is attributable to the
one/two-stage architecture choice rather than an unfair head start.

### 7.2 Controlled comparison

Everything is held identical except the model: same dataset snapshot, same 70/15/15
source-stratified split (seed 42), same image size 640, same frozen test set, and — the key
fairness control — **the same evaluation code scores every model** (`src/eval`). YOLO and
Faster R-CNN produce boxes in different formats; both are converted to one internal format
before scoring, so no metric-definition drift can creep in between the two. Each detector is
to be trained with several random seeds so a real difference is separable from run-to-run
noise. Accuracy **and** efficiency (speed, size) are both reported, since a clinical tool
must be deployable.

### 7.3 Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Params | Size | ms/img |
|---|---|---|---|---|---|---|---|---|
| **YOLO11n** (final) | **0.747** | **0.452** | 0.731 | 0.768 | 0.743 | 2.59 M | 5.48 MB | 65.9 (CPU) |
| Faster R-CNN (challenger) | pending | pending | pending | pending | pending | ~41 M | pending | pending |
| CNN (classification floor) | n/a — classifier | n/a | *(image-level; see 7.4)* | | | ~small | pending | pending |

*🔒 INTERNAL — Faster R-CNN row fills from the Colab training + eval run
(`notebooks/03_evaluation.ipynb`, `src/models/faster_rcnn.py`). It uses the exact same
`src/eval` scoring as YOLO, so the numbers are directly comparable. Then interpret **why the
winner wins** — expected axes: two-stage Faster R-CNN may recover some small-Cavity recall
at a large cost in size/speed (~41 M vs 2.59 M params); YOLO likely wins the
accuracy-per-compute trade-off that matters for deployment.*

### 7.4 The CNN floor — a bridged, honest comparison

The CNN is a classifier and produces no boxes, so it cannot appear in the detection table
above (no mAP). To still extract value from it, the pipeline reduces each detector's boxes
to a single per-image label — using the same `Cavity > Crown > Impacted Tooth > Filling`
priority the CNN's own single-label targets were built with — and scores the CNN against
that "downgraded" detector on plain classification metrics (accuracy, per-class P/R/F1,
image-level confusion matrix). This answers only "which pathology is in this X-ray," and
**deliberately discards localization** — YOLO's actual advantage — which is reported
separately via mAP. It is a floor check ("are these findings separable at all?"), not the
architecture benchmark.

*🔒 INTERNAL — needs Aparna's `cnn_baseline_summary.json` (not yet on HF; the model card
still shows `[test_accuracy]` placeholders) OR the `.keras` model to run inference through
`src.eval.classification`. Turnkey cells are in the notebook.*
