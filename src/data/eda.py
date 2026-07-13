"""Loading helpers for EDA over the combined dataset produced by prepare_dataset.py."""

import os

import cv2
import pandas as pd

from src.data.config import DATA_PROCESSED, SPLITS, TARGET_CLASSES


def source_of(filename: str) -> str:
    return "xray1" if filename.startswith("xray1_") else "panoramic"


def load_annotations(dataset_root=DATA_PROCESSED) -> pd.DataFrame:
    dataset_root = str(dataset_root)
    rows = []
    for split in SPLITS:
        label_dir = os.path.join(dataset_root, split, "labels")
        image_dir = os.path.join(dataset_root, split, "images")
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            stem = fname[:-4]
            image_files = [f for f in os.listdir(image_dir) if f.startswith(stem + ".")]
            image_path = os.path.join(image_dir, image_files[0]) if image_files else None

            with open(os.path.join(label_dir, fname)) as f:
                lines = [l.split() for l in f if l.strip()]

            for parts in lines:
                cls_idx = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                rows.append({
                    "split": split, "source": source_of(stem), "image": stem,
                    "image_path": image_path, "class": TARGET_CLASSES[cls_idx],
                    "x_center": x, "y_center": y, "bbox_w": w, "bbox_h": h,
                    "bbox_area_norm": w * h,
                })

    return pd.DataFrame(rows)


def load_image_metadata(dataset_root=DATA_PROCESSED) -> pd.DataFrame:
    dataset_root = str(dataset_root)
    rows = []
    for split in SPLITS:
        image_dir = os.path.join(dataset_root, split, "images")
        if not os.path.isdir(image_dir):
            continue
        for fname in os.listdir(image_dir):
            path = os.path.join(image_dir, fname)
            img = cv2.imread(path)
            if img is None:
                rows.append({"split": split, "source": source_of(fname), "image": fname,
                             "width": None, "height": None, "corrupt": True})
                continue
            h, w = img.shape[:2]
            rows.append({"split": split, "source": source_of(fname), "image": fname,
                         "width": w, "height": h, "corrupt": False})
    return pd.DataFrame(rows)
