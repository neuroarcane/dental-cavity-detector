from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/"data"
MODELS_DIR = ROOT/"models"

CLASS_NAMES = ["Cavity", "Filling", "Crown", "Impacted Tooth"]
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
IMG_SIZE = 640
