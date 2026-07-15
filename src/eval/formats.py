"""Common detection/ground-truth format shared across model backends.

Both the YOLO (Ultralytics) and Faster R-CNN (torchvision) predictions get
converted to this format before scoring, so the rest of `src.eval` only has
to know one data shape regardless of which architecture produced the boxes -
that's what keeps the baseline-vs-challenger benchmark apples-to-apples.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class GroundTruth:
    box: List[float]  # [x1, y1, x2, y2] pixel coords
    class_id: int  # 0-indexed, matches src.config.CLASS_NAMES


@dataclass
class Detection:
    box: List[float]  # [x1, y1, x2, y2] pixel coords
    class_id: int  # 0-indexed, matches src.config.CLASS_NAMES
    score: float


GroundTruths = Dict[str, List[GroundTruth]]  # image_id -> ground-truth boxes
Detections = Dict[str, List[Detection]]  # image_id -> predicted boxes


def from_yolo_result(result) -> List[Detection]:
    """Convert a single Ultralytics `Results` object to a `Detection` list."""
    boxes = result.boxes
    if boxes is None:
        return []
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(box=[x1, y1, x2, y2], class_id=int(box.cls[0]), score=float(box.conf[0]))
        )
    return detections


def from_torchvision_output(output: dict, score_threshold: float = 0.0) -> List[Detection]:
    """Convert a single torchvision detection model output dict to a `Detection` list.

    torchvision detection models reserve label 0 for background and number
    real classes from 1, so class ids are shifted back to 0-indexed here to
    match `src.config.CLASS_NAMES`.
    """
    boxes = output["boxes"].tolist()
    labels = output["labels"].tolist()
    scores = output["scores"].tolist()
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        detections.append(Detection(box=list(box), class_id=int(label) - 1, score=float(score)))
    return detections


def to_coco_gt(ground_truths: GroundTruths, class_names: Sequence[str]) -> Tuple[dict, Dict[str, int]]:
    """Build a COCO-format ground-truth dict for `pycocotools.COCOeval`.

    Returns the COCO dict plus the image_id -> integer-id map used to align
    predictions with it in `to_coco_detections`.
    """
    images, annotations = [], []
    ann_id = 1
    image_id_map = {image_id: idx + 1 for idx, image_id in enumerate(ground_truths.keys())}

    for image_id, img_idx in image_id_map.items():
        images.append({"id": img_idx, "file_name": image_id})
        for gt in ground_truths[image_id]:
            x1, y1, x2, y2 = gt.box
            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": gt.class_id + 1,  # COCO category ids are 1-indexed
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}, image_id_map


def to_coco_detections(detections: Detections, image_id_map: Dict[str, int]) -> List[dict]:
    """Build a COCO-format results list for `pycocotools.COCOeval`."""
    results = []
    for image_id, dets in detections.items():
        img_idx = image_id_map.get(image_id)
        if img_idx is None:
            continue
        for det in dets:
            x1, y1, x2, y2 = det.box
            results.append({
                "image_id": img_idx,
                "category_id": det.class_id + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": det.score,
            })
    return results
