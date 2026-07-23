"""Default hyperparameter search spaces for the YOLO sweep.

`GRID_SEARCH_SPACE`: a small, exhaustive set for a quick first pass, limited
to the parameters most likely to matter here - learning rate, plus the two
augmentation flags worth sanity-checking on X-ray images specifically (mosaic
can paste unrelated anatomy together in ways that don't occur in real
panoramic X-rays; horizontal flip is anatomically fine since jaws are
left-right symmetric).

`RANDOM_SEARCH_SPACE`: a wider set to sample from once the grid pass has
narrowed down promising regions.
"""
import random
from typing import Any, Callable, Dict, List, Union

Distribution = Union[List[Any], Callable[[random.Random], Any]]

GRID_SEARCH_SPACE: Dict[str, list] = {
    "lr0": [0.001, 0.01, 0.02],
    "mosaic": [0.0, 1.0],
    "fliplr": [0.0, 0.5],
}

RANDOM_SEARCH_SPACE: Dict[str, Distribution] = {
    "lr0": [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05],
    "lrf": [0.01, 0.1, 0.2],
    "momentum": [0.85, 0.9, 0.937, 0.95],
    "weight_decay": [0.0, 0.0005, 0.001],
    "warmup_epochs": [0.0, 3.0, 5.0],
    "mosaic": [0.0, 0.5, 1.0],
    "mixup": [0.0, 0.1, 0.2],
    "hsv_h": [0.0, 0.015, 0.03],
    "hsv_s": [0.4, 0.7],
    "hsv_v": lambda rng: round(rng.uniform(0.2, 0.6), 2),
    "degrees": [0.0, 5.0, 10.0],
    "translate": [0.0, 0.1, 0.2],
    "scale": [0.3, 0.5, 0.9],
    "fliplr": [0.0, 0.5],
}


def sample_random_params(distributions: Dict[str, Distribution], rng: random.Random) -> Dict[str, Any]:
    return {key: (dist(rng) if callable(dist) else rng.choice(dist)) for key, dist in distributions.items()}
