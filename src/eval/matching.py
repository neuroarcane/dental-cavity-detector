"""Greedy IoU-based matching between ground-truth and predicted boxes.

Two matching modes are used by the rest of `src.eval`:
- class-aware: a prediction must share the GT's class to count as a match
  (used for precision/recall/F1, since a correct-location wrong-class
  prediction should not count as a true positive).
- class-agnostic: predictions match GT by location only, and the class
  comparison happens afterwards (used for the confusion matrix, so
  misclassifications show up instead of being silently dropped as an
  FP+FN pair).
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


@dataclass
class MatchResult:
    matched: List[Tuple[int, int, float]]  # (gt_idx, det_idx, iou)
    unmatched_gt: List[int]
    unmatched_det: List[int]


def greedy_match(
    gt_boxes: Sequence[Sequence[float]],
    gt_classes: Sequence[int],
    det_boxes: Sequence[Sequence[float]],
    det_classes: Sequence[int],
    det_scores: Sequence[float],
    iou_threshold: float,
    class_agnostic: bool = False,
) -> MatchResult:
    """Match detections to ground truth, highest-confidence detection first."""
    det_order = sorted(range(len(det_boxes)), key=lambda i: det_scores[i], reverse=True)
    gt_taken = [False] * len(gt_boxes)
    matched: List[Tuple[int, int, float]] = []
    unmatched_det: List[int] = []

    for di in det_order:
        best_iou, best_gi = 0.0, -1
        for gi in range(len(gt_boxes)):
            if gt_taken[gi]:
                continue
            if not class_agnostic and gt_classes[gi] != det_classes[di]:
                continue
            iou = iou_xyxy(gt_boxes[gi], det_boxes[di])
            if iou >= iou_threshold and iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            gt_taken[best_gi] = True
            matched.append((best_gi, di, best_iou))
        else:
            unmatched_det.append(di)

    unmatched_gt = [gi for gi, taken in enumerate(gt_taken) if not taken]
    return MatchResult(matched=matched, unmatched_gt=unmatched_gt, unmatched_det=unmatched_det)
