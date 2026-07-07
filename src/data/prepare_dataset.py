"""
Merge the two X-ray object-detection sources into one YOLO-bbox dataset scoped to
the sprint's 4-class list: Cavity, Filling, Crown, Impacted Tooth.

Sources (X-ray only -- the intraoral-photo "Cavity Dataset" is intentionally
excluded, see docs/README.md for the reasoning):
    A: data/raw/Dental X-ray.v1i.yolov11/            (axis-aligned YOLO bbox labels)
    B: data/raw/Dental X-Ray Panoramic Dataset/YOLO/YOLO/  (YOLO segmentation-polygon labels, 31 classes)

Output:
    data/processed/{train,valid,test}/{images,labels}/
    data/processed/data.yaml

Only object instances belonging to the 4 target classes are kept. Polygons from
the panoramic source are converted to axis-aligned boxes (min/max of the polygon
points). Images left with zero target-class instances after filtering are
dropped (not copied) so the combined set doesn't fill up with unlabeled
background images pulled in from the 31-class source.

Usage:
    python -m src.data.prepare_dataset
"""

import os
import shutil
import zipfile
from pathlib import Path

from src.data.config import (
    CLASS_MAP_A,
    CLASS_MAP_B,
    COLAB_DRIVE_PROJECT_FOLDER,
    DATA_PROCESSED,
    SOURCE_A_NAME,
    SOURCE_B_SUBPATH,
    SPLITS,
    TARGET_CLASSES,
    TARGET_INDEX,
    data_raw_dir,
    is_colab,
)


def ensure_colab_data_extracted() -> None:
    """On Colab, copy+extract the dataset zips from Drive into local disk once per session."""
    data_root = data_raw_dir()
    data_root.mkdir(parents=True, exist_ok=True)

    zip_names = ["Dental X-ray.v1i.yolov11.zip", "Dental X-Ray Panoramic Dataset.zip"]
    for zip_name in zip_names:
        zip_path = os.path.join(COLAB_DRIVE_PROJECT_FOLDER, zip_name)
        marker = data_root / (zip_name + ".extracted")
        if marker.exists():
            print(f"Already extracted: {zip_name}")
            continue
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"Expected {zip_path}\n"
                "Upload the dataset zip to that Drive folder first (see data/README.md)."
            )
        print(f"Extracting {zip_name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_root)
        marker.touch()
        print(f"Done: {zip_name}")


def read_yolo_names(data_yaml_path):
    """Minimal YAML parser for the `names:` block (avoids a pyyaml dependency)."""
    names = {}
    in_names_block = False
    with open(data_yaml_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                inline = stripped[len("names:"):].strip()
                if inline.startswith("["):
                    items = [s.strip().strip("'\"") for s in inline.strip("[]").split(",")]
                    return {i: n for i, n in enumerate(items)}
                in_names_block = True
                continue
            if in_names_block:
                if not stripped or ":" not in line:
                    break
                idx_str, name = line.split(":", 1)
                names[int(idx_str.strip())] = name.strip()
    return names


def build_index_map(names_by_idx, class_map):
    """source_class_idx -> target_class_idx (or None if dropped)."""
    lower_map = {k.lower(): v for k, v in class_map.items()}
    result = {}
    for idx, name in names_by_idx.items():
        target_name = lower_map.get(name.strip().lower())
        result[idx] = TARGET_INDEX[target_name] if target_name else None
    return result


def convert_label_line_bbox(parts, index_map):
    """Source A: `class x y w h` (already axis-aligned) -> re-map class only."""
    src_idx = int(parts[0])
    target_idx = index_map.get(src_idx)
    if target_idx is None:
        return None
    return f"{target_idx} {' '.join(parts[1:5])}"


def convert_label_line_polygon(parts, index_map):
    """Source B: `class x1 y1 x2 y2 ... xn yn` polygon -> axis-aligned bbox."""
    src_idx = int(parts[0])
    target_idx = index_map.get(src_idx)
    if target_idx is None:
        return None

    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return f"{target_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def process_source(source_root, prefix, index_map, line_converter, output_root, stats):
    for split in SPLITS:
        image_dir = os.path.join(source_root, split, "images")
        label_dir = os.path.join(source_root, split, "labels")
        if not os.path.isdir(image_dir):
            continue

        out_image_dir = os.path.join(output_root, split, "images")
        out_label_dir = os.path.join(output_root, split, "labels")
        os.makedirs(out_image_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        for image_name in os.listdir(image_dir):
            stem, ext = os.path.splitext(image_name)
            label_path = os.path.join(label_dir, stem + ".txt")
            if not os.path.isfile(label_path):
                continue

            with open(label_path) as f:
                raw_lines = [l.split() for l in f if l.strip()]

            kept_lines = []
            for parts in raw_lines:
                converted = line_converter(parts, index_map)
                if converted is not None:
                    kept_lines.append(converted)

            stats["images_seen"][split] += 1
            if not kept_lines:
                stats["images_dropped_empty"][split] += 1
                continue

            new_stem = f"{prefix}_{stem}"
            shutil.copyfile(os.path.join(image_dir, image_name), os.path.join(out_image_dir, new_stem + ext))
            with open(os.path.join(out_label_dir, new_stem + ".txt"), "w") as f:
                f.write("\n".join(kept_lines) + "\n")

            stats["images_kept"][split] += 1
            for line in kept_lines:
                cls_idx = int(line.split()[0])
                stats["instance_counts"][TARGET_CLASSES[cls_idx]] += 1


def merge_datasets(output_root: Path = DATA_PROCESSED) -> dict:
    """Build the combined dataset and return merge stats."""
    if is_colab():
        ensure_colab_data_extracted()

    data_root = data_raw_dir()
    source_a = os.path.join(data_root, SOURCE_A_NAME)
    source_b = os.path.join(data_root, SOURCE_B_SUBPATH)

    output_root = str(output_root)
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)

    names_a = read_yolo_names(os.path.join(source_a, "data.yaml"))
    names_b = read_yolo_names(os.path.join(source_b, "data.yaml"))
    index_map_a = build_index_map(names_a, CLASS_MAP_A)
    index_map_b = build_index_map(names_b, CLASS_MAP_B)

    stats = {
        "images_seen": {s: 0 for s in SPLITS},
        "images_kept": {s: 0 for s in SPLITS},
        "images_dropped_empty": {s: 0 for s in SPLITS},
        "instance_counts": {c: 0 for c in TARGET_CLASSES},
        "classes_kept_a": {names_a[i]: TARGET_CLASSES[t] for i, t in index_map_a.items() if t is not None},
        "classes_kept_b": {names_b[i]: TARGET_CLASSES[t] for i, t in index_map_b.items() if t is not None},
    }

    process_source(source_a, "xray1", index_map_a, convert_label_line_bbox, output_root, stats)
    process_source(source_b, "panoramic", index_map_b, convert_label_line_polygon, output_root, stats)

    with open(os.path.join(output_root, "data.yaml"), "w") as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {TARGET_CLASSES}\n")

    return stats


def print_merge_summary(stats: dict) -> None:
    print("Source A classes kept:", stats["classes_kept_a"])
    print("Source B classes kept:", stats["classes_kept_b"])
    print()
    for split in SPLITS:
        print(f"{split}: seen={stats['images_seen'][split]} "
              f"kept={stats['images_kept'][split]} "
              f"dropped(no target-class objects)={stats['images_dropped_empty'][split]}")
    print("\nInstance counts (target classes only):")
    for cls, count in stats["instance_counts"].items():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    merge_stats = merge_datasets()
    print_merge_summary(merge_stats)
    print(f"\nCombined dataset written to: {DATA_PROCESSED}")
