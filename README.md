# Dental Cavity Detector

AASD 4014 - Deep Learning II final project. Multi-class object detection on dental X-rays using YOLO transfer learning.

## Team

| Name | Role |
|---|---|
| Ali | Project Manager / Scrum Lead |
| Varsha | Data Lead |
| Aparna | Model Lead |
| Temirlan | Tuning / Benchmarking Lead |
| Iva | Evaluation & Report Lead |

## Problem

Detect and localize dental pathologies on panoramic X-rays with bounding boxes across four classes: Cavity, Filling, Crown, and Impacted Tooth.

## Repo Structure

```
data/            datasets (not committed - see data/README.md)
notebooks/       EDA, training, and evaluation notebooks
src/             reusable Python code (data, models, eval)
models/          trained checkpoints (not committed)
docs/            labeling guidelines and other project docs
```

## Setup

```
pip install -r requirements.txt
```

## Links

Jira board: https://neuroarcane.atlassian.net/jira/software/projects/SCRUM/boards/1
Confluence docs: https://neuroarcane.atlassian.net/wiki/spaces/DCD/overview
Report skeleton & metrics spec: https://neuroarcane.atlassian.net/wiki/spaces/DCD/pages/655420
