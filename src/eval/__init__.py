from src.eval.classification import (
    compute_classification_metrics,
    image_labels_from_detections,
    image_labels_from_ground_truth,
)
from src.eval.confusion import compute_confusion_matrix
from src.eval.formats import (
    Detection,
    GroundTruth,
    from_torchvision_output,
    from_yolo_result,
)
from src.eval.loaders import load_ground_truth
from src.eval.metrics import compute_map, compute_pr_curve, compute_precision_recall_f1
from src.eval.predict import predict_torchvision_split, predict_yolo_split
from src.eval.report import build_detection_report, render_samples
from src.eval.visualize import draw_predictions, plot_confusion_matrix, plot_pr_curve

__all__ = [
    # detection metrics
    "compute_map",
    "compute_pr_curve",
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    # classification metrics (CNN baseline)
    "compute_classification_metrics",
    "image_labels_from_ground_truth",
    "image_labels_from_detections",
    # plotting
    "plot_confusion_matrix",
    "plot_pr_curve",
    "draw_predictions",
    # formats + IO
    "Detection",
    "GroundTruth",
    "from_yolo_result",
    "from_torchvision_output",
    "load_ground_truth",
    "predict_yolo_split",
    "predict_torchvision_split",
    # report orchestration
    "build_detection_report",
    "render_samples",
]
