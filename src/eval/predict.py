"""Run a trained detector over a `data/processed` split into the shared `src.eval` format.

One function per backend (Ultralytics YOLO/RT-DETR vs. torchvision Faster R-CNN)
since their inference APIs differ, but both return the same `Detections` shape
so `src.eval.metrics` / `src.eval.confusion` can score either one identically.
"""
from pathlib import Path

from src.data.config import DATA_PROCESSED
from src.eval.formats import Detections, from_torchvision_output, from_yolo_result


def predict_yolo_split(model, split: str, dataset_root=DATA_PROCESSED, **predict_kwargs) -> Detections:
    """Run an Ultralytics `YOLO`/`RTDETR` model over every image in `split`."""
    image_dir = Path(dataset_root) / split / "images"
    detections: Detections = {}
    for result in model.predict(source=str(image_dir), stream=True, verbose=False, **predict_kwargs):
        image_id = Path(result.path).name
        detections[image_id] = from_yolo_result(result)
    return detections


def predict_torchvision_split(
    model, split: str, dataset_root=DATA_PROCESSED, device: str = "cpu", score_threshold: float = 0.0
) -> Detections:
    """Run a torchvision detection model (e.g. Faster R-CNN) over every image in `split`."""
    import cv2
    import torch

    image_dir = Path(dataset_root) / split / "images"
    detections: Detections = {}
    model.eval()
    with torch.no_grad():
        for image_path in sorted(image_dir.iterdir()):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            output = model(tensor)[0]
            detections[image_path.name] = from_torchvision_output(output, score_threshold=score_threshold)
    return detections
