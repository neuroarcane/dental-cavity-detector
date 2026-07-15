"""mAP (via pycocotools) and fixed-threshold precision/recall/F1."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.formats import Detections, GroundTruths, to_coco_detections, to_coco_gt
from src.eval.matching import greedy_match


def _run_coco_eval(
    ground_truths: GroundTruths,
    detections: Detections,
    class_names: Sequence[str],
    iou_thrs: Optional[Sequence[float]] = None,
) -> Optional[COCOeval]:
    """Run COCOeval and return it, or None if there are no detections to score."""
    coco_gt_dict, image_id_map = to_coco_gt(ground_truths, class_names)
    coco_dt_list = to_coco_detections(detections, image_id_map)
    if not coco_dt_list:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        gt_path.write_text(json.dumps(coco_gt_dict))
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(coco_dt_list)

        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            if iou_thrs is not None:
                coco_eval.params.iouThrs = np.array(iou_thrs)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
    return coco_eval


def compute_map(ground_truths: GroundTruths, detections: Detections, class_names: Sequence[str]) -> dict:
    """Overall + per-class mAP@0.5 and mAP@0.5:0.95 via COCOeval."""
    coco_eval = _run_coco_eval(ground_truths, detections, class_names)
    if coco_eval is None:
        return {"map50": 0.0, "map50_95": 0.0, "per_class_ap50": {name: 0.0 for name in class_names}}

    per_class_ap50 = {}
    for k, name in enumerate(class_names):
        # index 0 = IoU 0.5, COCOeval's default first threshold.
        precision = coco_eval.eval["precision"][0, :, k, 0, -1]
        precision = precision[precision > -1]
        per_class_ap50[name] = float(precision.mean()) if precision.size else float("nan")

    return {
        "map50_95": float(coco_eval.stats[0]),
        "map50": float(coco_eval.stats[1]),
        "per_class_ap50": per_class_ap50,
    }


def compute_pr_curve(
    ground_truths: GroundTruths,
    detections: Detections,
    class_names: Sequence[str],
    class_name: str,
    iou_threshold: float = 0.5,
):
    """Precision values across the 101 standard COCO recall thresholds for one class."""
    coco_eval = _run_coco_eval(ground_truths, detections, class_names, iou_thrs=[iou_threshold])
    if coco_eval is None:
        return [0.0], [0.0]

    k = list(class_names).index(class_name)
    precision = coco_eval.eval["precision"][0, :, k, 0, -1]
    recalls = coco_eval.params.recThrs.tolist()
    return recalls, [float(p) if p > -1 else 0.0 for p in precision.tolist()]


def compute_precision_recall_f1(
    ground_truths: GroundTruths,
    detections: Detections,
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> dict:
    """Per-class + macro-averaged precision/recall/F1 at one fixed operating point.

    Unlike `compute_map` (which integrates over all confidence thresholds),
    this scores a single confidence cutoff - the number the report needs for
    a plain precision/recall/F1 table.
    """
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in class_names}

    for image_id, gts in ground_truths.items():
        dets = [d for d in detections.get(image_id, []) if d.score >= score_threshold]
        gt_boxes = [g.box for g in gts]
        gt_classes = [g.class_id for g in gts]
        det_boxes = [d.box for d in dets]
        det_classes = [d.class_id for d in dets]
        det_scores = [d.score for d in dets]

        match = greedy_match(
            gt_boxes, gt_classes, det_boxes, det_classes, det_scores, iou_threshold, class_agnostic=False
        )

        for gi, _di, _iou in match.matched:
            per_class[class_names[gt_classes[gi]]]["tp"] += 1
        for gi in match.unmatched_gt:
            per_class[class_names[gt_classes[gi]]]["fn"] += 1
        for di in match.unmatched_det:
            per_class[class_names[det_classes[di]]]["fp"] += 1

    results = {}
    for name, counts in per_class.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[name] = {"precision": precision, "recall": recall, "f1": f1, **counts}

    macro = {
        metric: sum(r[metric] for r in results.values()) / len(results)
        for metric in ("precision", "recall", "f1")
    }
    return {"per_class": results, "macro": macro}
