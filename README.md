# Dental Cavity Detector

Multi-class object detection on dental X-rays (Cavity, Filling, Crown, Impacted Tooth) via YOLO transfer learning — now being extended into a deployed, full-stack system with a business case.

**Trained models:** https://huggingface.co/aparnamohankumar/dental-cavity-detector

## Project phases

- **AASD 4014 — Deep Learning II** (complete): built and evaluated the detector. Overall mAP@0.5 = 0.747.
- **AASD 4016 — Full Stack Data Science Systems** (in progress): deploy the trained model end-to-end — Flask API → Docker → cloud → dashboard → API-as-a-service.
- **AASD 4017 — Presenting Data Science-driven Solutions** (in progress): build and pitch a business case for the solution.

Planning, tasks, and docs for the current phase live in Jira and the new Confluence space (links below). **Team rule: all work is tracked in both Jira and GitHub** so any task can be picked up by another member if needed.

## Team

| Name | Role |
|---|---|
| Ali | Project Lead / Scrum Master |
| Hessam | Deployment / MLOps Lead — Flask API, Docker, cloud, API service |
| Aparna | Model Lead — model packaging & serving (+ business research) |
| Varsha | Data / Dashboard Lead — inference, client app, dashboard (+ business research) |
| Iva | Business Lead — the AASD 4017 business case & pitch |

_Temirlan (Tuning / Benchmarking / Evaluation Lead for the DL II phase) completed his work — merged to `main` — and has since left the team._

## Problem

Detect and localize dental pathologies on dental X-rays with bounding boxes across four classes: Cavity, Filling, Crown, and Impacted Tooth. The tool is framed as assistive — a clinician "second read" — rather than autonomous diagnosis.

## Models

| Role | Model | Notes |
|---|---|---|
| Final detector | YOLO11n (COCO-pretrained, transfer learning) | 50 epochs; overall mAP@0.5 = 0.747 |
| Baseline | CNN trained from scratch | Single-label classifier — no localization, so it produces no mAP |
| Challenger | Faster R-CNN (two-stage CNN) | Detector-vs-detector benchmark |

### Results (YOLO11n, held-out test set — src/eval pipeline, conf 0.25)

| Class | AP@0.5 | Recall |
|---|---|---|
| Impacted Tooth | 0.937 | 0.960 |
| Crown | 0.867 | 0.944 |
| Filling | 0.738 | 0.733 |
| Cavity | 0.445 | 0.435 |

**Known limitation:** performance is inversely related to clinical importance. Cavity — the most clinically consequential class — is the weakest (recall 0.435, i.e. roughly 6 in 10 real cavities are missed). Cavities are small, low-contrast radiolucencies, exactly what a detector pretrained on natural images struggles with. Raising Cavity recall is the primary tuning objective.

## Dataset

Two X-ray sources merged into a single YOLO-bbox dataset scoped to the four target classes. An intraoral colour-photo dataset was deliberately excluded (different imaging modality). Preprocessing: MD5 deduplication → source-stratified 70/15/15 re-split → train-only oversampling. Validation and test splits are left at the true distribution. Fixed seed = 42, image size = 640.

**Data ethics:** the panoramic source embeds real patient names in image filenames (identifiable health information). No patient identifier was ever committed to this repository (raw/processed data are gitignored; notebooks hold only aggregate charts). The pipeline hashes filename stems so no name propagates into processed data or prediction outputs.

## Repo structure

```
data/          datasets (not committed — see data/README.md)
notebooks/     01_eda, 02_preprocessing, 03_evaluation
src/           reusable Python code (data, models, eval, tuning)
models/        trained checkpoints (not committed) + runs/
docs/          project docs
api/           [4016] Flask serving API (in progress)
dashboard/     [4016] dashboard connected to the deployed model (in progress)
deployment/    [4016] Dockerfile, cloud deploy config & notes (in progress)
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

- **Jira board:** https://neuroarcane.atlassian.net/jira/software/projects/SCRUM/boards/1
- **Confluence (Full Stack & Business Case):** https://neuroarcane.atlassian.net/wiki/spaces/DCDFSBC
- **Confluence (DL II project):** https://neuroarcane.atlassian.net/wiki/spaces/DCD/overview
- **Trained models:** https://huggingface.co/aparnamohankumar/dental-cavity-detector
