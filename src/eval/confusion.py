"""Detection confusion matrix: class-agnostic box matching + class comparison."""
from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.eval.formats import Detections, GroundTruths
from src.eval.matching import greedy_match

BACKGROUND = "background"


def compute_confusion_matrix(
    ground_truths: GroundTruths,
    detections: Detections,
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> Tuple[np.ndarray, list]:
    """Returns (matrix, labels) where labels = list(class_names) + ['background'].

    Boxes are matched by location only (class-agnostic), then the matched
    pair's classes are compared - this surfaces misclassifications (e.g. a
    Filling predicted as a Crown) instead of silently counting them as one
    false positive plus one false negative, the way `compute_precision_recall_f1`
    would.
    """
    labels = list(class_names) + [BACKGROUND]
    y_true, y_pred = [], []

    for image_id, gts in ground_truths.items():
        dets = [d for d in detections.get(image_id, []) if d.score >= score_threshold]
        gt_boxes = [g.box for g in gts]
        gt_classes = [g.class_id for g in gts]
        det_boxes = [d.box for d in dets]
        det_classes = [d.class_id for d in dets]
        det_scores = [d.score for d in dets]

        match = greedy_match(
            gt_boxes, gt_classes, det_boxes, det_classes, det_scores, iou_threshold, class_agnostic=True
        )

        for gi, di, _iou in match.matched:
            y_true.append(class_names[gt_classes[gi]])
            y_pred.append(class_names[det_classes[di]])
        for gi in match.unmatched_gt:
            y_true.append(class_names[gt_classes[gi]])
            y_pred.append(BACKGROUND)
        for di in match.unmatched_det:
            y_true.append(BACKGROUND)
            y_pred.append(class_names[det_classes[di]])

    matrix = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return matrix, labels
