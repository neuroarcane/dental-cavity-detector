"""Orchestration: turn a model's detections into the full benchmark artifact set.

Framework-agnostic - it takes an already-computed `Detections` dict, so the same
code produces the report for YOLO and for the Faster R-CNN challenger. Emits a
metrics JSON, confusion-matrix / PR-curve / per-class-AP plots, and a set of
qualitative sample panels.

PHI note: source images are real de-identified clinical X-rays whose *filenames*
still carry patient names on this branch (the Section 3.8 hashing patch is not
applied here). Sample panels are therefore written with generic names
(sample_01.png ...) and captions.json never records the source filename.
"""
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.confusion import compute_confusion_matrix
from src.eval.formats import Detections, GroundTruths
from src.eval.metrics import compute_map, compute_pr_curve, compute_precision_recall_f1
from src.eval.visualize import draw_predictions, plot_confusion_matrix, plot_pr_curve


def build_detection_report(
    detections: Detections,
    ground_truth: GroundTruths,
    class_names: Sequence[str],
    out_dir,
    model_name: str,
    efficiency: Optional[dict] = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> dict:
    """Compute all detection metrics + save plots and metrics.json. Returns the metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_res = compute_map(ground_truth, detections, class_names)
    prf = compute_precision_recall_f1(ground_truth, detections, class_names, iou_threshold, score_threshold)
    cm, labels = compute_confusion_matrix(ground_truth, detections, class_names, iou_threshold, score_threshold)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(cm, labels, normalize=True, ax=ax)
    ax.set_title(f"{model_name} - confusion matrix (row-normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # PR curves (all classes on one axis)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in class_names:
        recalls, precisions = compute_pr_curve(ground_truth, detections, class_names, name, iou_threshold)
        plot_pr_curve(recalls, precisions, label=f"{name} (AP={map_res['per_class_ap50'][name]:.3f})", ax=ax)
    ax.set_title(f"{model_name} - PR curves @ IoU {iou_threshold}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_pr_curves.png", dpi=150)
    plt.close(fig)

    # Per-class AP bar
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(class_names)
    ax.bar(names, [map_res["per_class_ap50"][n] for n in names], color="#4C72B0")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} - per-class AP@0.5")
    for i, n in enumerate(names):
        ax.text(i, map_res["per_class_ap50"][n] + 0.02, f"{map_res['per_class_ap50'][n]:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_per_class_ap.png", dpi=150)
    plt.close(fig)

    metrics = {
        "model": model_name,
        "map50": map_res["map50"],
        "map50_95": map_res["map50_95"],
        "per_class_ap50": map_res["per_class_ap50"],
        "precision_recall_f1": prf,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "efficiency": efficiency or {},
        "settings": {"iou_threshold": iou_threshold, "score_threshold": score_threshold},
        "n_test_images": len(ground_truth),
    }
    (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _missed_class(gt_boxes, dets, class_id, class_names, iou_threshold=0.5, score_threshold=0.25):
    """True if any ground-truth box of `class_id` has no matching same-class detection (a miss)."""
    from src.eval.matching import greedy_match

    kept = [d for d in dets if d.score >= score_threshold]
    match = greedy_match(
        [g.box for g in gt_boxes], [g.class_id for g in gt_boxes],
        [d.box for d in kept], [d.class_id for d in kept], [d.score for d in kept],
        iou_threshold, class_agnostic=False,
    )
    missed_gt_classes = {gt_boxes[gi].class_id for gi in match.unmatched_gt}
    return class_id in missed_gt_classes


def render_samples(
    detections: Detections,
    ground_truth: GroundTruths,
    images_dir,
    class_names: Sequence[str],
    out_dir,
    n_good: int = 5,
    n_failure: int = 3,
    failure_class: str = "Cavity",
    score_threshold: float = 0.25,
) -> list:
    """Save qualitative GT-vs-prediction panels with PHI-safe generic names.

    Picks `n_failure` images where `failure_class` ground truth is missed (the
    clinically important error mode) and `n_good` images with several correct
    findings. Returns caption records (no source filenames).
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)
    image_by_stem = {p.stem: p for p in images_dir.iterdir()}
    failure_id = class_names.index(failure_class)

    failures, goods = [], []
    for image_id, gts in ground_truth.items():
        dets = detections.get(image_id, [])
        gt_classes = {g.class_id for g in gts}
        if failure_id in gt_classes and _missed_class(gts, dets, failure_id, class_names, score_threshold=score_threshold):
            failures.append(image_id)
        elif len(gt_classes) >= 3:
            goods.append(image_id)

    chosen = [("failure", i) for i in failures[:n_failure]] + [("good", i) for i in goods[:n_good]]
    captions = []
    for idx, (kind, image_id) in enumerate(chosen, 1):
        stem = Path(image_id).stem
        image_path = image_by_stem.get(stem)
        if image_path is None:
            continue
        gts = ground_truth[image_id]
        dets = [d for d in detections.get(image_id, []) if d.score >= score_threshold]

        # left = ground truth (green), right = predictions (red)
        gt_img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        for g in gts:
            x1, y1, x2, y2 = [int(v) for v in g.box]
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(gt_img, class_names[g.class_id], (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        pred_img = draw_predictions(str(image_path), dets, class_names, score_threshold)

        h = max(gt_img.shape[0], pred_img.shape[0])
        panel = cv2.hconcat([cv2.resize(gt_img, (int(gt_img.shape[1] * h / gt_img.shape[0]), h)),
                             cv2.resize(pred_img, (int(pred_img.shape[1] * h / pred_img.shape[0]), h))])
        out_name = f"sample_{idx:02d}_{kind}.png"
        cv2.imwrite(str(out_dir / out_name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        captions.append({
            "file": out_name, "kind": kind,
            "gt_classes": sorted({class_names[g.class_id] for g in gts}),
            "pred_classes": sorted({class_names[d.class_id] for d in dets}),
            "note": ("Missed " + failure_class + " (false negative)") if kind == "failure"
                    else "Multiple correct findings",
        })

    (out_dir / "captions.json").write_text(json.dumps(captions, indent=2))
    return captions
