# docs

Project documentation for the dental cavity detector: labeling guidelines, class definitions, and process notes. Narrative docs and specs live here; live task tracking stays in Jira and Confluence (linked from the root README).

## Labeling Guidelines

Annotations are bounding boxes in YOLO format across four classes:

| Class ID | Class | Definition |
| --- | --- | --- |
| 0 | Cavity | Radiolucent decay in enamel or dentin; box the lesion, not the whole tooth. |
| 1 | Filling | Existing restorative material (radiopaque); box the restoration. |
| 2 | Crown | Full-coverage restoration over a tooth; box the crown outline. |
| 3 | Impacted Tooth | Tooth failing to erupt into normal position; box the full impacted tooth. |

## Conventions

- One box per finding; overlapping findings get separate boxes.
- Box tightly around the visible extent of the feature.
- When uncertain between classes, flag for review rather than guessing.
- Keep label files paired one-to-one with images (same basename).

## Contents

| File | Purpose |
| --- | --- |
| `labeling-guidelines.md` | Full annotation protocol and edge cases (to be added). |
| `class-definitions.md` | Reference images and boundary criteria per class (to be added). |
