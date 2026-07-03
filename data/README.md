# Data

Datasets are not committed to this repo. Source and prep steps:

## Sources

Roboflow and Kaggle dental X-ray datasets (DENTEX-based sets preferred, since they already cover multiple pathology classes). See SCRUM-1 in Jira for the finalized source list.

## Classes

Cavity, Filling, Crown, Impacted Tooth (4 classes, not binary).

## Structure (local only, gitignored)

```
data/raw/            original downloaded datasets
data/processed/      cleaned + consolidated dataset
data/annotations/    YOLO-format label files
```
