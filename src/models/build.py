from ultralytics import YOLO
from src.config import NUM_CLASSES

def build_baseline(weights: str = "yolo8n.pt") -> YOLO:
  """Load a COCO-pretrained YOLO checkpoint for transfer learning."""
  return YOLO(weights)
