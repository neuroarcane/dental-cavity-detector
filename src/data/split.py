"""
Deduplication and stratified re-splitting for data/processed/.

prepare_dataset.py's merge just concatenates each source dataset's own
train/valid/test split. That's a problem: EDA (notebooks/01_eda.ipynb) found the
two sources are ~13:1 imbalanced in image count, and the smaller source
(xray1) contributes zero Crown labels -- so evaluating on the naive merged
test split mostly measures performance on the panoramic source.

This module pools every kept image across the naive splits (ignoring which
split it originally came from), drops exact-duplicate images, then reassigns
images to train/valid/test stratified by source dataset, so every split gets
proportional representation from both sources.

Note this fixes *source* imbalance, not the overall Filling >> Cavity/Crown
*class* imbalance -- that's a separate problem best handled at training time
(class-weighted loss / oversampling), not by splitting differently.
"""

import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.config import DATA_PROCESSED, SPLITS


def _file_hash(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_exact_duplicates(dataset_root=DATA_PROCESSED) -> dict:
    """Drop exact-duplicate images (by content hash) across all splits, keeping
    the first occurrence. Returns {"checked": N, "removed": [stems]}."""
    dataset_root = Path(dataset_root)
    seen_hashes = {}
    removed = []
    checked = 0

    for split in SPLITS:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.is_dir():
            continue

        for image_path in sorted(image_dir.iterdir()):
            checked += 1
            digest = _file_hash(str(image_path))
            if digest in seen_hashes:
                stem = image_path.stem
                image_path.unlink()
                label_path = label_dir / (stem + ".txt")
                if label_path.exists():
                    label_path.unlink()
                removed.append(stem)
            else:
                seen_hashes[digest] = str(image_path)

    return {"checked": checked, "removed": removed}


def pool_images(dataset_root=DATA_PROCESSED) -> pd.DataFrame:
    """List every remaining image across all splits with its current split and source."""
    dataset_root = Path(dataset_root)
    rows = []
    for split in SPLITS:
        image_dir = dataset_root / split / "images"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            stem = image_path.stem
            source = "xray1" if stem.startswith("xray1_") else "panoramic"
            rows.append({
                "stem": stem, "ext": image_path.suffix,
                "current_split": split, "source": source,
            })
    return pd.DataFrame(rows)


def stratified_resplit(
    dataset_root=DATA_PROCESSED,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Pool all images (ignoring their current split) and reassign to
    train/valid/test stratified by source dataset, then move files accordingly."""
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-9

    dataset_root = Path(dataset_root)
    pool = pool_images(dataset_root)

    train_stems, rest = train_test_split(
        pool, train_size=train_frac, stratify=pool["source"], random_state=random_state,
    )
    valid_stems, test_stems = train_test_split(
        rest, train_size=valid_frac / (valid_frac + test_frac),
        stratify=rest["source"], random_state=random_state,
    )

    new_split_by_stem = {}
    for df, split in [(train_stems, "train"), (valid_stems, "valid"), (test_stems, "test")]:
        for _, row in df.iterrows():
            new_split_by_stem[row["stem"]] = (split, row["current_split"], row["ext"])

    moved = 0
    for stem, (new_split, old_split, ext) in new_split_by_stem.items():
        if new_split == old_split:
            continue

        old_image = dataset_root / old_split / "images" / (stem + ext)
        old_label = dataset_root / old_split / "labels" / (stem + ".txt")
        new_image_dir = dataset_root / new_split / "images"
        new_label_dir = dataset_root / new_split / "labels"
        new_image_dir.mkdir(parents=True, exist_ok=True)
        new_label_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(old_image), str(new_image_dir / (stem + ext)))
        if old_label.exists():
            shutil.move(str(old_label), str(new_label_dir / (stem + ".txt")))
        moved += 1

    return {
        "total_images": len(pool),
        "moved": moved,
        "split_counts": {
            "train": len(train_stems), "valid": len(valid_stems), "test": len(test_stems),
        },
        "split_source_counts": {
            "train": train_stems["source"].value_counts().to_dict(),
            "valid": valid_stems["source"].value_counts().to_dict(),
            "test": test_stems["source"].value_counts().to_dict(),
        },
    }
