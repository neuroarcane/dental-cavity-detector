"""Faster R-CNN (ResNet-50-FPN) challenger detector - torchvision, COCO-pretrained.

This is the second-architecture challenger from docs/benchmarking-methodology.md.
It is NOT Ultralytics-native, so it needs its own dataset adapter and training
loop here; its predictions are converted back to the shared `src.eval` format
via `from_torchvision_output`, so it is scored by the exact same metrics code as
the YOLO baseline.

Convention: torchvision detection models reserve label 0 for background and
number real classes from 1, so this module emits/consumes 1-indexed labels.
`src.eval.formats.from_torchvision_output` shifts them back to the 0-indexed
ids used everywhere else.

Real training runs on Colab GPU; locally this is CPU-only and meant for
scaffold checks, not convergence.
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import NUM_CLASSES, SEED
from src.data.config import DATA_PROCESSED


def build_faster_rcnn(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """COCO-pretrained Faster R-CNN with the box head resized to our classes (+background)."""
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # +1 for the background class that torchvision detectors require at index 0.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


class YoloDetectionDataset(Dataset):
    """Reads a `data/processed` split (YOLO-format labels) as torchvision detection targets.

    Labels come in normalized `class xc yc w h`; they are converted to
    pixel-space xyxy boxes and 1-indexed class labels, which is what
    torchvision detection models expect as targets.
    """

    def __init__(self, split: str, dataset_root=DATA_PROCESSED):
        dataset_root = Path(dataset_root)
        self.image_dir = dataset_root / split / "images"
        self.label_dir = dataset_root / split / "labels"
        image_by_stem = {p.stem: p for p in self.image_dir.iterdir()}
        # Keep only labels that have a matching image on disk.
        self.samples = [
            (image_by_stem[lbl.stem], lbl)
            for lbl in sorted(self.label_dir.glob("*.txt"))
            if lbl.stem in image_by_stem
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_path = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_idx, xc, yc, w, h = line.split()
                xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                # Drop degenerate boxes (zero area) - torchvision's loss errors on them.
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_idx) + 1)  # shift to 1-indexed (0 = background)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return tensor, target


def collate_fn(batch: List[Tuple]) -> Tuple[tuple, tuple]:
    """Detection batches hold variable box counts per image, so keep them as tuples."""
    images, targets = zip(*batch)
    return images, targets


def train_faster_rcnn(
    split_train: str = "train",
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    device: str = None,
    num_workers: int = 2,
    seed: int = SEED,
    max_batches: int = None,
    save_path: str = None,
):
    """Fine-tune Faster R-CNN on the processed dataset.

    Returns ``(model, history)`` where ``history["epoch_loss"]`` is the per-epoch
    mean loss. If ``save_path`` is given, the trained weights are saved there (via
    ``state_dict``); reload with ``build_faster_rcnn()`` + ``load_state_dict``.

    `max_batches` caps batches per epoch - used only for a fast scaffold check;
    leave it None for real training. Run the full training on Colab GPU per the
    benchmark methodology (matched epoch budget + multiple seeds vs. YOLO).
    """
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = YoloDetectionDataset(split_train)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    model = build_faster_rcnn().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    history = {"epoch_loss": []}
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for i, (images, targets) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # returns losses in train mode
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            n += 1
        history["epoch_loss"].append(running / max(n, 1))

    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return model, history
