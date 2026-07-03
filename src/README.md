# src

Reusable Python code for the dental cavity detection pipeline. Notebooks in `notebooks/` import from here, so shared logic lives in this package rather than in notebook cells.

## Modules

| Module | Responsibility |
| --- | --- |
| `data/` | Dataset loading, YOLO-format label parsing, train/val/test splits, and augmentation. |
| `models/` | Model definition and transfer-learning setup for the YOLO detector. |
| `eval/` | Metrics (mAP, precision/recall), confusion matrix, and prediction visualization. |

## Conventions

- Each subpackage has an `__init__.py` so code imports as `src.data`, `src.models`, `src.eval`.
- Keep functions pure and testable; avoid hardcoded paths by passing config or arguments.
- Shared constants (class names, dataset paths) live in a single config module.
