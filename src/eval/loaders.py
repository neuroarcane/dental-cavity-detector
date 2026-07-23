"""Load ground truth for a `data/processed` split into the shared `src.eval` format.

Label files are YOLO format (normalized `class x_center y_center w h`), so
each box is converted to pixel-space xyxy using the paired image's actual
dimensions - that's the coordinate space `src.eval.formats` and predictions
from both YOLO and torchvision detectors use.
"""
from pathlib import Path

import cv2

from src.data.config import DATA_PROCESSED
from src.eval.formats import GroundTruth, GroundTruths


def load_ground_truth(split: str, dataset_root=DATA_PROCESSED) -> GroundTruths:
    """Read YOLO-format labels for one split (train/valid/test).

    Keyed by image filename, which is what `predict_yolo_split` /
    `predict_torchvision_split` use as `image_id` so predictions line up with
    ground truth by key.
    """
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_by_stem = {p.stem: p for p in image_dir.iterdir()}
    ground_truths: GroundTruths = {}

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_by_stem.get(label_path.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        boxes = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_idx, xc, yc, w, h = line.split()
                cls_idx = int(cls_idx)
                xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
                boxes.append(GroundTruth(
                    box=[xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2],
                    class_id=cls_idx,
                ))
        ground_truths[image_path.name] = boxes

    return ground_truths
