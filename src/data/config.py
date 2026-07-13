"""Shared constants and paths for the dental cavity detection data pipeline."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# Colab: data lives on local disk (copied+extracted from Drive), not the cloned repo.
COLAB_DATA_ROOT = Path("/content/dental_data")
COLAB_DRIVE_PROJECT_FOLDER = "/content/drive/MyDrive/Dental Cavity Detection"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def data_raw_dir() -> Path:
    return COLAB_DATA_ROOT if is_colab() else DATA_RAW


SOURCE_A_NAME = "Dental X-ray.v1i.yolov11"
SOURCE_B_SUBPATH = os.path.join("Dental X-Ray Panoramic Dataset", "YOLO", "YOLO")

TARGET_CLASSES = ["Cavity", "Filling", "Crown", "Impacted Tooth"]
TARGET_INDEX = {name: i for i, name in enumerate(TARGET_CLASSES)}

# source class name -> target class name
CLASS_MAP_A = {
    "Cavity": "Cavity",
    "Fillings": "Filling",
    "Impacted Tooth": "Impacted Tooth",
    "Implant": None,  # not in target list
}
CLASS_MAP_B = {
    "Caries": "Cavity",
    "Crown": "Crown",
    "Filling": "Filling",
    "impacted tooth": "Impacted Tooth",
    # everything else in the 31-class panoramic taxonomy maps to None (dropped)
}

SPLITS = ["train", "valid", "test"]
