# Second-Architecture Shortlist & Benchmark Methodology

**Owner:** Temirlan (Tuning / Benchmarking Lead)
**Jira task:** Shortlist second-architecture candidate + define benchmark methodology
**Status:** Draft for team / instructor sign-off
**Related rubric sections:** Report §4 Model Description, §6 Hyperparameter Tuning (architecture exploration), §7 Benchmarking

---

## 1. Purpose & scope

This document (a) selects the **second detector architecture** we will train as a
challenger to our YOLO baseline, and (b) defines the **benchmarking protocol** we will
use to compare them fairly. It does **not** cover running the training itself — that
is a downstream task. The goal here is a spec the whole team can build against so the
final "benchmark vs. baseline" analysis is fair, reproducible, and defensible in the report.

**Detection task recap:** multi-class object detection on dental panoramic X-rays.
Classes: `cavity`, `filling`, `crown`, `impacted_tooth`. Labels in YOLO format
(Roboflow/Kaggle sourced). Baseline = YOLOv11/v12 transfer learning (Ultralytics).

### 1.1 The three models we compare

The benchmark spans **three** models in **two task families**. This is deliberate: it
covers both the "compared to a baseline" and the "architecture exploration" rubric points.

| Tier | Model | Task | Role |
|---|---|---|---|
| Simple baseline | CNN classifier (3 conv blocks, from scratch) — Aparna's HF push | **Classification** (one label/image) | "What does a naive approach get?" |
| Primary | YOLOv11n (Ultralytics, COCO-pretrained) | **Detection** (boxes) | The product |
| Challenger | Faster R-CNN (ResNet-50-FPN, COCO-pretrained) | **Detection** (boxes) | Architecture comparison vs. YOLO |

**Two metric families, because the tasks differ:**
- **Detection** (YOLO vs. Faster R-CNN) → mAP, IoU box-matching, box-level confusion
  matrix. This is the core apples-to-apples architecture benchmark. Code: `src.eval`
  (`compute_map`, `compute_precision_recall_f1`, `compute_confusion_matrix`).
- **Classification** (CNN baseline) → accuracy, per-class P/R/F1, image-level confusion
  matrix. mAP is **undefined** for a classifier (no boxes, no IoU). Code:
  `src.eval.classification`.
- **Bridging the two:** to place the CNN and YOLO on one axis, a detector's boxes are
  reduced to a single per-image label (`image_labels_from_detections`) using the same
  `Cavity > Crown > Impacted Tooth > Filling` priority rule Aparna used to make her
  training set single-label. Both models then answer the same "which pathology is in this
  X-ray" question. This intentionally discards YOLO's localization — that strength is
  reported separately via mAP, not hidden.

---

## 2. Baseline recap (what we benchmark against)

| Property | Value |
|---|---|
| Model | YOLOv11/v12 (Ultralytics), COCO-pretrained, fine-tuned on our 4 classes |
| Type | Single-stage, anchor-free CNN detector, NMS post-processing |
| Input size | 640 × 640 (`imgsz=640`) |
| Framework | Ultralytics `YOLO` API |
| Primary metric | mAP@0.5, mAP@0.5:0.95 |

> The baseline is the reference point. The challenger must be scored by the **same eval
> code** on the **same held-out test set** for the comparison to be valid.

---

## 3. Candidate architectures considered

| Candidate | Paradigm | Same framework? | Contrast value | Integration risk | Verdict |
|---|---|---|---|---|---|
| **Faster R-CNN** | Two-stage CNN (region proposals + ROI head) | No — torchvision | High (1-stage → 2-stage CNN) | Medium (COCO-format conversion) | **Selected** |
| RetinaNet | One-stage CNN + focal loss | No — torchvision | Medium (imbalance-friendly) | Medium | Alternative |
| RT-DETR (rtdetr-l) | Transformer enc-dec, NMS-free | Yes — Ultralytics `RTDETR` | High (CNN → transformer) | Low | Rejected — team decided to keep the comparison CNN-vs-CNN, not introduce a transformer |
| YOLOv8 / older YOLO | Same as baseline | Yes | Low — too similar | Low | Rejected — weak benchmark |
| DETR / Deformable DETR | Transformer | No | High | High — slow convergence, data-hungry | Rejected — timeline risk |
| EfficientDet | BiFPN, compound scaling | No — TF | Medium | High | Rejected — framework friction |

### 3.1 Recommendation: **Faster R-CNN (ResNet-50-FPN, COCO-pretrained)**

*Updated per team discussion 2026-07-13: team preferred a CNN-vs-CNN comparison over the
transformer contrast, so the pick moved from RT-DETR to Faster R-CNN (previously our
documented fallback).*

**Why:**
- **Meaningful architectural contrast within CNNs.** Faster R-CNN is a two-stage detector
  (region proposal network → ROI head) versus YOLO's single-stage anchor-free design. This
  is a classic, well-understood axis of comparison (speed/simplicity vs. proposal-based
  accuracy) and is easy to explain and justify in the report — satisfies "architecture
  exploration" (rubric §6) without the added risk of a transformer's training dynamics.
- **Relevant to our data.** Two-stage detectors are traditionally stronger on small/dense
  objects — plausibly relevant here since `cavity` lesions can be small relative to the
  full X-ray. Worth calling out explicitly in the Discussion section regardless of which
  way the result goes.
- **COCO-pretrained** `torchvision.models.detection.fasterrcnn_resnet50_fpn` weights are
  available → same transfer-learning story as the baseline.
- **Integration cost is real but contained.** Unlike YOLO, Faster R-CNN is not
  Ultralytics-native, so: (a) YOLO-format labels must be converted to COCO-format for
  training/eval, and (b) predictions must be adapted into a shared internal format before
  scoring. Both of these are now handled centrally by the `src/eval` package (see
  `src/eval/formats.py`), so the benchmark stays apples-to-apples: **one shared metrics
  implementation scores both models**, regardless of which framework produced the boxes.

**Alternative kept on the shelf — RetinaNet:** a one-stage CNN with focal loss, useful if
Faster R-CNN's training time turns out to be too expensive for our compute budget; keeps
the CNN-only constraint while being closer to YOLO's single-stage family.

**Rejected — RT-DETR:** still the lower-integration-risk option technically (same
Ultralytics API as baseline), but the team's explicit preference is to compare two CNNs
rather than introduce a transformer, so it's dropped from consideration for this benchmark.

**Decision needed from team/instructor:** confirm Faster R-CNN (ResNet-50-FPN) as the
challenger (§7).

---

## 4. Benchmark methodology

**Guiding principle:** *hold everything constant except the architecture.* Any variable we
don't control becomes a confound that invalidates the comparison.

### 4.1 Controlled variables (fairness protocol)

| Variable | Setting (identical for both models) |
|---|---|
| Dataset | Same consolidated dataset, same version/snapshot |
| Train / val / test split | **Same split, same fixed seed** — test set frozen before any tuning |
| Image size | `imgsz=640` for both |
| Pretraining | Both COCO-pretrained → fine-tuned (transfer learning) |
| Training budget | Same max epochs + **same early-stopping patience** on val mAP |
| Augmentation | Same policy where the framework allows; any unavoidable difference documented |
| Hardware | Same GPU for all timed runs (record model, VRAM, batch size) |
| Eval code | **One shared eval harness** (`src/eval`) scores both models |
| Confidence / IoU thresholds | Same values for reported P/R/F1 and for NMS-style filtering |

> **Golden rule:** the test set is touched **only once**, at the end, for final reported
> numbers. All tuning and model selection happen on the validation set.

### 4.2 Metrics

**Primary (accuracy):**
- mAP@0.5 and mAP@0.5:0.95 (COCO-style), overall + **per class**
- Precision, Recall, F1 at a fixed confidence threshold
- Precision–Recall curves (overall and per class)
- Confusion matrix (per class, incl. background/false-positive rows)

**Secondary (efficiency / deployability — YOLO is real-time, so this matters):**
- Parameter count and GFLOPs
- Inference latency (ms/image) at fixed batch size & hardware; report FPS
- Model size on disk (MB)
- Total training time to best epoch

**Why efficiency counts:** a fair benchmark isn't "who has the highest mAP" — it's the
accuracy/latency/size trade-off. This directly feeds the Discussion (§8 of the report).

### 4.3 Experimental protocol

1. Freeze dataset snapshot and split (seed logged in `config.py`).
2. Train baseline (YOLO) and challenger (RT-DETR) under the controlled settings above.
3. **Repeat each training with ≥3 seeds** and report **mean ± std** so a difference is
   distinguishable from run-to-run noise. (If GPU time is tight, minimum: report variance
   from the seeds we can afford and state the limitation.)
4. Select best checkpoint per model by **validation** mAP@0.5:0.95.
5. Run the frozen **test set once** through the shared eval harness for both models.
6. Log everything (config, seed, metrics, curves) per run for reproducibility.

### 4.4 Evaluation-harness contract

Both models must be scored by identical code in `src/eval` so no metric-definition drift
creeps in. The harness takes `(model_predictions, ground_truth)` and returns the metric set
in §4.2. For RT-DETR this is native (Ultralytics `.val()`); for a non-Ultralytics fallback
(Faster R-CNN) predictions are adapted to the same interface before scoring.

### 4.5 Statistical treatment

- Report **mean ± std** across seeds for every headline metric.
- Treat a difference as meaningful only if it exceeds the seed-to-seed spread.
- Note per-class results explicitly — dental class imbalance (e.g. `cavity` vs `crown`)
  means overall mAP can hide large per-class gaps.

### 4.6 Deliverable artifacts (for the report §5, §7)

- **Benchmark table:** both models × all metrics in §4.2 (mean ± std).
- **PR curves** overlaid, baseline vs challenger.
- **Per-class AP** grouped bar chart.
- **Confusion matrices**, side by side.
- **Qualitative panel:** same 5–10 test X-rays, both models' predictions side by side
  (satisfies the "5–10 sample outputs" submission requirement too).
- **Accuracy-vs-latency** scatter to visualize the trade-off.

### 4.7 Threats to validity (state these in the report)

- **Framework differences** in default augmentation/loss — Faster R-CNN is torchvision,
  not Ultralytics, so its default augmentation pipeline differs from YOLO's. We mitigate
  this by aligning what we can (resize, horizontal flip, normalization) and explicitly
  documenting what we can't (mosaic/mixup are YOLO-specific and have no Faster R-CNN
  equivalent) rather than silently ignoring the gap.
- **Small test set** → wide confidence intervals; mitigated by multi-seed reporting.
- **Class imbalance** → always report per-class, not just aggregate mAP.
- **Pretraining domain gap** (COCO natural images → grayscale X-rays) affects both models
  equally, so it doesn't bias the *comparison*.

---

## 5. Definition of done (this task)

- [ ] Challenger architecture selected and signed off (RT-DETR-L proposed).
- [ ] Controlled-variable table agreed with Model Lead & Eval Lead.
- [ ] Metric set + shared eval-harness contract agreed with Eval Lead.
- [ ] Artifact list agreed with Report Lead.
- [ ] This doc committed to `docs/` and mirrored to Confluence.

## 6. Rubric mapping

| Rubric item | Covered by |
|---|---|
| §4 Model Description | §2, §3 (two justified architectures) |
| §6 Hyperparameter Tuning / architecture exploration | §3 selection rationale + tuning downstream |
| §7 Benchmarking vs baseline | §4 entire methodology |
| "Innovative + justified model choice" (top tier) | Faster R-CNN two-stage contrast, §3.1 |
| §5 visual results / confusion matrix | §4.6 artifacts |
| 5–10 sample outputs (submission) | §4.6 qualitative panel |

## 7. Open decisions (need sign-off)

1. **Confirm Faster R-CNN (ResNet-50-FPN)** as challenger (vs. RetinaNet alternative).
2. **Confirm baseline YOLO version** (v11 vs v12) so both are pinned in `requirements.txt`.
3. **Seed budget:** how many seeds per model can our GPU time afford (target ≥3)?
4. **Fixed thresholds:** agree the confidence + IoU values for P/R/F1 reporting.
