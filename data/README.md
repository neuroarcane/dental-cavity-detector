# Data

## Sources

Two X-ray Roboflow exports, merged down to the 4-class list below (`src/data/prepare_dataset.py`):
- `Dental X-ray.v1i.yolov11` — axis-aligned YOLO bbox labels
- `Dental X-Ray Panoramic Dataset` — YOLO segmentation-polygon labels, converted to bboxes on merge

An intraoral-photo Kaggle dataset was evaluated and intentionally excluded (different imaging
modality — see git history / PR discussion for the reasoning). See SCRUM-1 in Jira for the
full source evaluation.

**Privacy note**: filenames in the panoramic source embed real patient names (e.g.
`SALEHI_KOBRA_2020-07-12184001`). Keep that in mind before uploading any of this data to a
third-party service (e.g. don't use cloud-hosted annotation tools on it).

## Classes

Cavity, Filling, Crown, Impacted Tooth (4 classes, not binary). Class order/index is fixed by
`src/data/config.py:TARGET_CLASSES` and mirrored in `data/annotations/classes.txt` — don't
reorder either without updating the other.

## Structure

```
data/raw/                   committed to git (~1GB, all files well under GitHub's 100MB limit,
                             no LFS needed) — the two source datasets as downloaded
data/processed/              gitignored, regenerated locally by running
                             notebooks/01_eda.ipynb then notebooks/02_preprocessing.ipynb
data/annotations/classes.txt committed — class list for LabelImg, order matches TARGET_CLASSES
data/annotations/to_label/   gitignored — local scratch space for new images awaiting manual
                             annotation (see "Manual annotation" below)
```

## Manual annotation (LabelImg)

For labeling additional images beyond the two merged sources (e.g. filling gaps in an
under-represented class):

1. `pip install labelImg`
2. Drop new unlabeled images into `data/annotations/to_label/images/`
3. Launch: `labelImg data/annotations/to_label/images data/annotations/classes.txt data/annotations/to_label/labels`
4. In LabelImg: set save format to **YOLO** (button on the left toolbar — it cycles
   PascalVOC/YOLO/CreateML), then annotate. Boxes save as one `.txt` per image directly into
   `to_label/labels/`, already in the right format and class order to merge into
   `data/processed/` later.
5. Follow `docs/README.md`'s labeling guidelines (one box per finding, box tightly, flag
   uncertain cases rather than guessing).

`to_label/` is gitignored — newly annotated images aren't committed automatically. Decide
per-batch whether they should be (same size/privacy considerations as the raw datasets above).
