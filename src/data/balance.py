"""
Class-imbalance handling for data/processed/train/ via oversampling.

EDA (notebooks/01_eda.ipynb) found Filling instances (~54k) outnumber Cavity/Crown
(~11k each) roughly 5:1 in the merged dataset. This duplicates train-split images
containing under-represented classes so the model sees them more often per epoch.

Only ever touches the train split -- valid/test must stay untouched so evaluation
reflects the true, unmodified class distribution. Must run after stratified_resplit()
(src/data/split.py), not before, so duplicated images don't get shuffled into valid/test.

This is not the only way to handle class imbalance -- class-weighted loss is another
common approach -- but oversampling is the one that fits cleanly into a data-prep
pipeline without needing to own the training loop.
"""

import random
import shutil
from collections import Counter
from pathlib import Path

from src.data.config import DATA_PROCESSED, TARGET_CLASSES


def compute_class_instance_counts(dataset_root=DATA_PROCESSED, split: str = "train") -> dict:
    """Per-class object-instance counts in one split."""
    label_dir = Path(dataset_root) / split / "labels"
    counts = {c: 0 for c in TARGET_CLASSES}
    for label_path in label_dir.glob("*.txt"):
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[TARGET_CLASSES[int(line.split()[0])]] += 1
    return counts


def _index_split(dataset_root, split):
    """stem -> (extension, Counter of class_idx -> instance count in that image).

    Must be a per-instance count, not just a set of classes present: some images
    have several boxes of the same class, and duplicating them adds all of those
    real instances to disk, not just one per class.
    """
    image_dir = Path(dataset_root) / split / "images"
    label_dir = Path(dataset_root) / split / "labels"

    ext_by_stem = {p.stem: p.suffix for p in image_dir.iterdir()}
    index = {}
    for label_path in label_dir.glob("*.txt"):
        stem = label_path.stem
        if stem not in ext_by_stem:
            continue
        with open(label_path) as f:
            class_counts = Counter(int(line.split()[0]) for line in f if line.strip())
        index[stem] = (ext_by_stem[stem], class_counts)
    return index


def oversample_minority_classes(
    dataset_root=DATA_PROCESSED,
    split: str = "train",
    target_ratio: float = 0.5,
    max_duplicates_per_image: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Duplicate `split`-only images containing under-represented classes until each
    class reaches at least `target_ratio` of the majority class's instance count,
    or until any single source image has been duplicated `max_duplicates_per_image`
    times total (shared across classes -- an image boosting both Cavity and Crown
    still only gets duplicated up to the cap once, not once per class).
    """
    rng = random.Random(random_state)
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    index = _index_split(dataset_root, split)
    counts = compute_class_instance_counts(dataset_root, split)
    before = dict(counts)
    majority_count = max(counts.values())
    target = majority_count * target_ratio

    duplicated_images = {c: 0 for c in TARGET_CLASSES}
    dup_round_by_stem = {stem: 0 for stem in index}

    for cls_idx, class_name in enumerate(TARGET_CLASSES):
        if counts[class_name] >= target:
            continue

        candidates = [stem for stem, (_, class_counts) in index.items() if cls_idx in class_counts]
        if not candidates:
            continue
        rng.shuffle(candidates)

        # Multiple passes over candidates: each pass adds at most one duplicate per
        # image, so an image can be revisited on the next pass up to the shared cap.
        while counts[class_name] < target:
            made_progress = False
            for stem in candidates:
                if counts[class_name] >= target:
                    break
                if dup_round_by_stem[stem] >= max_duplicates_per_image:
                    continue

                dup_round_by_stem[stem] += 1
                ext, class_counts = index[stem]
                new_stem = f"{stem}_dup{dup_round_by_stem[stem]}"
                shutil.copyfile(image_dir / (stem + ext), image_dir / (new_stem + ext))
                shutil.copyfile(label_dir / (stem + ".txt"), label_dir / (new_stem + ".txt"))

                for present_idx, instance_count in class_counts.items():
                    counts[TARGET_CLASSES[present_idx]] += instance_count
                duplicated_images[class_name] += 1
                made_progress = True

            if not made_progress:
                break  # every candidate hit the per-image cap; stop even if target not reached

    return {
        "before": before,
        "after": counts,
        "duplicated_images": duplicated_images,
        "majority_class": max(before, key=before.get),
        "target_ratio": target_ratio,
    }
