# src

Reusable Python code for the dental cavity detection pipeline. Notebooks in `notebooks/` import from here, so shared logic lives in this package rather than in notebook cells.

## Modules

| Module | Responsibility |
| --- | --- |
| `data/` | Dataset loading, YOLO-format label parsing, train/val/test splits, and augmentation. |
| `models/` | Transfer-learning setup for the YOLO detector (`build.py`) and the Faster R-CNN challenger detector (`faster_rcnn.py`). |
| `eval/` | Framework-agnostic **detection** metrics (mAP, P/R/F1, confusion matrix) scoring YOLO or torchvision predictions through one shared format, plus **classification** metrics (`classification.py`) for the CNN baseline. See `docs/benchmarking-methodology.md` for how the three models are compared. |
| `tuning/` | Hyperparameter sweep runner (grid/random search over learning rate, augmentation, etc.) for the YOLO trainer. |

## Conventions

- Each subpackage has an `__init__.py` so code imports as `src.data`, `src.models`, `src.eval`.
- Keep functions pure and testable; avoid hardcoded paths by passing config or arguments.
- Shared constants (class names, dataset paths) live in a single config module.
