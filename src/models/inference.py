"""
src/models/inference.py — Dental Cavity Detector inference module (SCRUM-36)

Wraps the trained YOLO11n checkpoint for serving: loads weights once,
preprocesses an uploaded X-ray, runs inference, and returns structured
predictions (boxes + labels + confidence).

Follows repo conventions from src/README.md: imports shared config from
src.config, sits in src/models/ alongside build.py.

DoD: importable predict() function that runs on a sample image and
returns structured predictions.

Usage:
    from src.models.inference import predict, load_model

    results = predict("path/to/xray.jpg")
    # or, if you already have a loaded model (e.g. in a Flask app):
    model = load_model()
    results = predict("path/to/xray.jpg", model=model)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.config import CLASS_NAMES, IMG_SIZE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Canonical model source: Hugging Face Hub, not a local runs/ path.
# Ultralytics nests local training output under an extra runs/detect/
# prefix (see repo history), which makes local paths unreliable across
# environments — HF is what the team actually pulls from for eval,
# dashboard demos, and now serving.
HF_REPO_ID = os.environ.get("HF_REPO_ID", "aparnamohankumar/dental-cavity-detector")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "yolo11_baseline_best.pt")

# Inference settings — 640x640 matches training (src.config.IMG_SIZE).
# Confidence threshold is a first pass; revisit once we've made progress
# on the Cavity recall gap (0.435 in the ML Canvas / SCRUM-35) — lowering
# this threshold trades precision for recall and is one lever to test.
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model(
    repo_id: str = HF_REPO_ID,
    filename: str = HF_MODEL_FILENAME,
) -> YOLO:
    """
    Download (cached locally by huggingface_hub) and load the trained
    YOLO11n checkpoint from the Hugging Face Hub.

    Cached via lru_cache so repeated calls (e.g. from a Flask route
    handling many requests) don't re-resolve the download every time.
    Call this once at app startup in the serving layer to fail fast on
    a missing/unreachable checkpoint before accepting traffic.

    Note: for a private HF repo, HF_TOKEN must be set in the environment
    (or the runtime must already be logged in via huggingface_hub.login).
    """
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(checkpoint_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    image_path: str,
    model: YOLO | None = None,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    """
    Run inference on a single dental X-ray and return structured predictions.

    Args:
        image_path: Path to the uploaded X-ray image (jpg/png).
        model: Optional pre-loaded YOLO model. If omitted, loads (and
            caches) the model from the Hugging Face Hub.
        conf: Confidence threshold — predictions below this are dropped.
        iou: IoU threshold used for non-max suppression.

    Returns:
        A dict shaped for direct JSON serialization by the API layer:
        {
            "image": "xray_001.jpg",
            "detections": [
                {
                    "class": "Cavity",
                    "confidence": 0.87,
                    "box": {"x1": 120.4, "y1": 88.1, "x2": 210.7, "y2": 175.3}
                },
                ...
            ],
            "count": 1
        }
    """
    if model is None:
        model = load_model()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'.")

    # Ultralytics handles resize/normalize internally via imgsz, matching
    # the 640x640 training pipeline (src.config.IMG_SIZE) — no separate
    # manual preprocessing step needed here.
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    # predict() with a single image path returns a list with one Results obj
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                "confidence": round(float(box.conf.item()), 4),
                "box": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1),
                },
            }
        )

    return {
        "image": os.path.basename(image_path),
        "detections": detections,
        "count": len(detections),
    }


# ---------------------------------------------------------------------------
# Manual smoke test — satisfies the DoD ("runs on a sample image")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample_image = sys.argv[1] if len(sys.argv) > 1 else "sample_xray.jpg"

    print(f"Loading model from HF repo '{HF_REPO_ID}' ({HF_MODEL_FILENAME}) ...")
    m = load_model()

    print(f"Running inference on '{sample_image}' ...")
    output = predict(sample_image, model=m)

    print(f"\nFound {output['count']} finding(s):")
    for d in output["detections"]:
        print(f"  - {d['class']} (conf={d['confidence']}) box={d['box']}")
