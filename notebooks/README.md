# Notebooks

Jupyter notebooks for the dental cavity detection pipeline. Notebooks are for exploration and reporting; reusable logic lives in `src/`.

## Contents

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Exploratory data analysis of the dental X-ray dataset: class balance, image dimensions, bounding-box distributions, and sample visualizations. |
| `02_preprocessing.ipynb` | Deduplication, source-stratified train/valid/test re-split, and train-only class-imbalance oversampling. Fixes gaps `01_eda.ipynb` surfaces. |
| `03_training.ipynb` | YOLO transfer-learning training runs, including data config, hyperparameters, and training curves. |
| `04_evaluation.ipynb` | Model evaluation: mAP, per-class precision/recall, confusion matrix, and qualitative prediction review. |

## Conventions

- Keep reusable code in `src/` and import it here so notebooks stay readable.
- Restart-and-run-all before committing so outputs are reproducible.
- Do not commit datasets or model checkpoints (see `data/README.md` and `models/README.md`).
