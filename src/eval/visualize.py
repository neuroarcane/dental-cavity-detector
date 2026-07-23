"""Plotting helpers for the benchmark report: PR curves, confusion matrix, prediction overlays."""
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(matrix: np.ndarray, labels: Sequence[str], normalize: bool = True, ax=None):
    ax = ax or plt.gca()
    data = matrix.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)
    sns.heatmap(
        data, annot=True, fmt=".2f" if normalize else "d",
        xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    return ax


def plot_pr_curve(recalls: Sequence[float], precisions: Sequence[float], label: str, ax=None):
    ax = ax or plt.gca()
    ax.plot(recalls, precisions, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax


def draw_predictions(image_path: str, detections: list, class_names: Sequence[str], score_threshold: float = 0.25):
    """Draw predicted boxes on an image for qualitative review (report sample-outputs panel)."""
    import cv2

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for det in detections:
        if det.score < score_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.box]
        label = f"{class_names[det.class_id]} {det.score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image
