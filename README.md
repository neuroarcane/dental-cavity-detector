# Dental Cavity Detector

AASD 4014 - Deep Learning II final project. Multi-class object detection on dental X-rays using YOLO transfer learning.

**Trained models:** https://huggingface.co/aparnamohankumar/dental-cavity-detector

## Team

| Name | Role |
|---|---|
| Ali | Project Manager / Scrum Lead |
| Varsha | Data Lead |
| Aparna | Model Lead |
| Temirlan | Tuning / Benchmarking + Evaluation Lead |
| Iva | Business & Commercialization / Voice of Customer |

## Problem

Detect and localize dental pathologies on dental X-rays with bounding boxes across four classes: Cavity, Filling, Crown, and Impacted Tooth. The tool is framed as assistive - a clinician "second read" - rather than autonomous diagnosis.

## Models

| Role | Model | Notes |
|---|---|---|
| Final detector | YOLO11 (COCO-pretrained, transfer learning) | 50 epochs; **overall mAP@0.5 = 0.746** |
| Baseline | CNN trained from scratch | Single-label **classifier** - no localization, so it produces no mAP |
| Challenger | Faster R-CNN (two-stage CNN) | Detector-vs-detector benchmark |

### Results (YOLO11, held-out test set)

| Class | mAP@0.5 | Recall |
|---|---|---|
| Impacted Tooth | 0.942 | 0.957 |
| Crown | 0.874 | 0.925 |
| Filling | 0.748 | 0.703 |
| Cavity | 0.420 | 0.391 |

**Known limitation:** performance is inversely related to clinical importance. Cavity - the most clinically consequential class - is the weakest (recall 0.391, i.e. roughly 6 in 10 real cavities are missed). Cavities are small, low-contrast radiolucencies, which is exactly what a detector pretrained on natural images struggles with. Raising Cavity recall is the primary tuning objective.

## Dataset

Two X-ray sources merged into a single YOLO-bbox dataset scoped to the four target classes. An intraoral colour-photo dataset was deliberately excluded (different imaging modality - mixing RGB photos with grayscale radiographs introduces domain shift).

Preprocessing pipeline: MD5 deduplication, then a **source-stratified 70/15/15 re-split** (the sources are ~13:1 imbalanced and one contributes no Crown labels), then **train-only oversampling** for class imbalance (~5:1, Filling vs Cavity/Crown). Validation and test splits are left untouched so evaluation reflects the true distribution.

Fixed **seed = 42** and **image size = 640** for reproducibility.

### Data ethics

The panoramic source embeds **real patient names in its image filenames**. This is identifiable health information even though the names are not burned into the pixels. No patient identifier was ever committed to this repository (raw and processed data are gitignored; notebooks contain only aggregate charts). The preprocessing pipeline hashes filename stems so no name propagates into processed data or prediction outputs.

## Repo Structure

```
data/            datasets (not committed - see data/README.md)
notebooks/       01_eda, 02_preprocessing, 03_training, 04_evaluation
src/             reusable Python code (data, models, eval)
models/          trained checkpoints (not committed) + runs/
docs/            labeling guidelines and other project docs
```

## Setup

```
pip install -r requirements.txt
```

Load the trained detector directly, without retraining:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
    repo_id="aparnamohankumar/dental-cavity-detector",
    filename="yolo11_baseline_best.pt",
)
model = YOLO(model_path)
```

## Links

Jira board: https://neuroarcane.atlassian.net/jira/software/projects/SCRUM/boards/1
Confluence docs: https://neuroarcane.atlassian.net/wiki/spaces/DCD/overview
Final report (working draft): https://neuroarcane.atlassian.net/wiki/spaces/DCD/pages/655420
Trained models: https://huggingface.co/aparnamohankumar/dental-cavity-detector
