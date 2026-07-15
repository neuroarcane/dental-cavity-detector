"""Hyperparameter sweep runner: grid or random search over YOLO train() overrides.

Decoupled from the actual training call: `run_grid_sweep` / `run_random_sweep`
take a `train_and_eval_fn(params) -> metrics dict` callback, so the
enumeration/bookkeeping logic here can be unit-tested with a stub function
and swapped to the real Ultralytics trainer (`train_yolo`, below) - or later a
Faster R-CNN trainer - without touching this module.
"""
import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List

import pandas as pd

from src.config import IMG_SIZE, SEED
from src.tuning.search_space import sample_random_params

TrainAndEvalFn = Callable[[Dict[str, Any]], Dict[str, float]]


@dataclass
class SweepTrial:
    params: Dict[str, Any]
    metrics: Dict[str, float]


def run_grid_sweep(train_and_eval_fn: TrainAndEvalFn, grid: Dict[str, list]) -> List[SweepTrial]:
    """Exhaustive sweep over every combination in `grid` (param -> list of values)."""
    keys = list(grid.keys())
    trials = []
    for combo in product(*grid.values()):
        params = dict(zip(keys, combo))
        trials.append(SweepTrial(params=params, metrics=train_and_eval_fn(params)))
    return trials


def run_random_sweep(
    train_and_eval_fn: TrainAndEvalFn,
    distributions: Dict[str, list],
    n_trials: int = 10,
    seed: int = SEED,
) -> List[SweepTrial]:
    """Randomized sweep: `n_trials` param sets sampled from `distributions`."""
    rng = random.Random(seed)
    trials = []
    for _ in range(n_trials):
        params = sample_random_params(distributions, rng)
        trials.append(SweepTrial(params=params, metrics=train_and_eval_fn(params)))
    return trials


def results_to_df(trials: List[SweepTrial], sort_by: str = "map50_95") -> pd.DataFrame:
    """Flatten trials into one row per trial, sorted best-first by `sort_by`."""
    rows = [{**trial.params, **trial.metrics} for trial in trials]
    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df


def train_yolo(
    weights: str,
    data_yaml: str,
    params: Dict[str, Any],
    epochs: int,
    project: str,
    name: str,
    seed: int = SEED,
    imgsz: int = IMG_SIZE,
) -> Dict[str, float]:
    """Train + validate one YOLO/RT-DETR trial; returns the metrics dict a sweep records.

    Kept short-epoch by design during search (pass a small `epochs`, e.g.
    10-15) - the point of a sweep is to rank configs relative to each other,
    not to fully converge every trial. Re-train the winning config for the
    full epoch budget separately, and with multiple seeds per the benchmark
    methodology (see docs/benchmarking-methodology.md).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, seed=seed,
        project=project, name=name, verbose=False, **params,
    )
    val_metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=False)
    return {
        "map50": float(val_metrics.box.map50),
        "map50_95": float(val_metrics.box.map),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
    }
