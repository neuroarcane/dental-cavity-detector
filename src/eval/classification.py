"""Image-level classification metrics, for scoring Aparna's CNN baseline.

The CNN is a classifier (one label per image), so the detection metrics in
`src.eval.metrics` (mAP, IoU matching) do not apply to it. This module scores
image-level predictions instead: accuracy, per-class precision/recall/F1, and
an image-level confusion matrix.

To compare the CNN against the YOLO detector on the same axis, a detector's
boxes are first reduced to a single per-image label (`image_labels_from_detections`)
using the same priority rule Aparna used to make her training set single-label,
so both models answer the same "which pathology is in this X-ray" question.
This deliberately discards YOLO's localization - that strength is reported
separately via mAP in `src.eval.metrics`.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.eval.formats import Detections, GroundTruths

# Aparna's multi-label -> single-label reduction priority (see her HF model notes).
DEFAULT_PRIORITY = ["Cavity", "Crown", "Impacted Tooth", "Filling"]

ImageLabels = Dict[str, int]  # image_id -> class_id


def reduce_class_ids_to_label(
    class_ids: Sequence[int], class_names: Sequence[str], priority: Optional[Sequence[str]] = None
) -> Optional[int]:
    """Collapse the class ids present in one image to a single label by priority.

    Returns None if `class_ids` is empty (no boxes / nothing to reduce).
    """
    if len(class_ids) == 0:
        return None
    priority = priority or DEFAULT_PRIORITY
    present = set(class_ids)
    for name in priority:
        idx = class_names.index(name)
        if idx in present:
            return idx
    # Any class not covered by the priority list: fall back to the lowest id present.
    return min(present)


def image_labels_from_ground_truth(
    ground_truths: GroundTruths, class_names: Sequence[str], priority: Optional[Sequence[str]] = None
) -> ImageLabels:
    """Per-image single label from detection ground truth (same reduction as the CNN's training set)."""
    labels: ImageLabels = {}
    for image_id, gts in ground_truths.items():
        label = reduce_class_ids_to_label([g.class_id for g in gts], class_names, priority)
        if label is not None:
            labels[image_id] = label
    return labels


def image_labels_from_detections(
    detections: Detections,
    class_names: Sequence[str],
    strategy: str = "top_score",
    priority: Optional[Sequence[str]] = None,
    score_threshold: float = 0.25,
) -> ImageLabels:
    """Reduce a detector's boxes to one label per image so it can be scored as a classifier.

    strategy:
        "top_score" - the class of the highest-confidence box (what the model is surest of).
        "priority"  - priority-rule reduction over all above-threshold boxes, matching
                      how the CNN's single-label ground truth is built.
    """
    labels: ImageLabels = {}
    for image_id, dets in detections.items():
        kept = [d for d in dets if d.score >= score_threshold]
        if not kept:
            continue
        if strategy == "top_score":
            labels[image_id] = max(kept, key=lambda d: d.score).class_id
        elif strategy == "priority":
            labels[image_id] = reduce_class_ids_to_label([d.class_id for d in kept], class_names, priority)
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")
    return labels


def compute_classification_metrics(
    y_true: ImageLabels, y_pred: ImageLabels, class_names: Sequence[str]
) -> dict:
    """Accuracy, per-class + macro precision/recall/F1, and confusion matrix.

    Only images present in both `y_true` and `y_pred` are scored; the count of
    images the model returned no prediction for is reported as `n_unpredicted`
    so a model that abstains isn't silently rewarded.
    """
    common = [img for img in y_true if img in y_pred]
    n_unpredicted = len(y_true) - len(common)

    true_ids = [y_true[img] for img in common]
    pred_ids = [y_pred[img] for img in common]
    label_indices = list(range(len(class_names)))

    matrix = sk_confusion_matrix(true_ids, pred_ids, labels=label_indices)
    accuracy = float(np.mean(np.array(true_ids) == np.array(pred_ids))) if common else 0.0

    per_class = {}
    for k, name in enumerate(class_names):
        tp = int(matrix[k, k])
        fp = int(matrix[:, k].sum() - tp)
        fn = int(matrix[k, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[k, :].sum())}

    macro = {
        metric: sum(r[metric] for r in per_class.values()) / len(per_class)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": macro,
        "confusion_matrix": matrix,
        "labels": list(class_names),
        "n_scored": len(common),
        "n_unpredicted": n_unpredicted,
    }
