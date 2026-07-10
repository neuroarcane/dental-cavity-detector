# Second-Architecture Shortlist & Benchmark Methodology

**Owner:** Temirlan (Tuning / Benchmarking Lead)
**Jira task:** Shortlist second-architecture candidate + define benchmark methodology
**Status:** Draft for team / instructor sign-off
**Related rubric sections:** Report §4 Model Description, §6 Hyperparameter Tuning (architecture exploration), §7 Benchmarking

---

## 1. Purpose & scope

This document (a) selects the **second detector architecture** we will train as a
challenger to our YOLO baseline, and (b) defines the **benchmarking protocol** we will
use to compare the two fairly. It does **not** cover running the training itself — that
is a downstream task. The goal here is a spec the whole team can build against so the
final "benchmark vs. baseline" analysis is fair, reproducible, and defensible in the report.

**Detection task recap:** multi-class object detection on dental panoramic X-rays.
Classes: `cavity`, `filling`, `crown`, `impacted_tooth`. Labels in YOLO format
(Roboflow/Kaggle sourced). Baseline = YOLOv11/v12 transfer learning (Ultralytics).

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
| **RT-DETR (rtdetr-l)** | Transformer enc-dec, **NMS-free** set prediction | Yes — Ultralytics `RTDETR` | High (CNN → transformer) | **Low** | **Selected** |
| Faster R-CNN | Two-stage CNN (region proposals + ROI head) | No — torchvision/Detectron2 | High (1-stage → 2-stage) | Medium (COCO-format conversion) | Fallback |
| RetinaNet | One-stage + focal loss | No — torchvision | Medium (imbalance-friendly) | Medium | Alternative |
| YOLOv8 / older YOLO | Same as baseline | Yes | Low — too similar | Low | Rejected — weak benchmark |
| DETR / Deformable DETR | Transformer | No | High | High — slow convergence, data-hungry | Rejected — timeline risk |
| EfficientDet | BiFPN, compound scaling | No — TF | Medium | High | Rejected — framework friction |

### 3.1 Recommendation: **RT-DETR-L**

**Why:**
- **Meaningful paradigm contrast.** RT-DETR is a transformer encoder-decoder that does
  NMS-free set prediction — architecturally the opposite of YOLO's anchor-free CNN + NMS
  pipeline. This makes "architecture exploration" (rubric §6) genuine rather than cosmetic.
- **Apples-to-apples, low risk.** RT-DETR ships inside the same Ultralytics package we
  already use. Same `data.yaml`, same `.train()` / `.val()` calls, **same mAP computation**.
  We change one class name (`YOLO` → `RTDETR`) and reuse the entire eval harness. No label
  reformatting, no second metrics implementation to reconcile.
- **Fair compute comparison.** RT-DETR-L is real-time and roughly parameter-comparable to a
  large YOLO model, so an accuracy *and* speed comparison is fair and interesting.
- **COCO-pretrained** weights available → same transfer-learning story as the baseline.

**Fallback — Faster R-CNN:** if the team prefers a classic two-stage *CNN* contrast, use
`torchvision` Faster R-CNN (ResNet-50-FPN, COCO-pretrained). Cost: convert YOLO labels →
COCO JSON and run eval through `pycocotools` / a shared adapter so numbers stay comparable.
Only take this path if we accept the extra integration work.

**Decision needed from team/instructor:** confirm RT-DETR-L as the challenger (§8).

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

- **Framework differences** in default augmentation/loss — minimized by picking RT-DETR
  (same framework); documented if the Faster R-CNN fallback is used.
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
| "Innovative + justified model choice" (top tier) | RT-DETR transformer contrast, §3.1 |
| §5 visual results / confusion matrix | §4.6 artifacts |
| 5–10 sample outputs (submission) | §4.6 qualitative panel |

## 7. Open decisions (need sign-off)

1. **Confirm RT-DETR-L** as challenger (vs. Faster R-CNN fallback).
2. **Confirm baseline YOLO version** (v11 vs v12) so both are pinned in `requirements.txt`.
3. **Seed budget:** how many seeds per model can our GPU time afford (target ≥3)?
4. **Fixed thresholds:** agree the confidence + IoU values for P/R/F1 reporting.
