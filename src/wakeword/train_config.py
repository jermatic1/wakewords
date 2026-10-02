"""Renders the microwakeword training YAML from a wakeword config."""

from pathlib import Path
from typing import Any, NamedTuple

import yaml

from wakeword.config import FEATURE_STEP_MS, WakewordConfig
from wakeword.paths import Paths


class NegativeSet(NamedTuple):
    sampling_weight: float
    truncation: str


# Pre-computed negative feature sets from the kahrendt/microwakeword dataset,
# with the sampling weights from the upstream training notebook.
NEGATIVE_SETS = {
    "speech": NegativeSet(10.0, "random"),
    "dinner_party": NegativeSet(10.0, "random"),
    "no_speech": NegativeSet(5.0, "random"),
    "dinner_party_eval": NegativeSet(0.0, "split"),  # validation and testing only
}


def _feature(path: Path, weight: float, truth: bool, truncation: str) -> dict[str, Any]:
    return {
        "features_dir": str(path.resolve()),
        "sampling_weight": weight,
        "penalty_weight": 1.0,
        "truth": truth,
        "truncation_strategy": truncation,
        "type": "mmap",
    }


def render(cfg: WakewordConfig, paths: Paths) -> dict[str, Any]:
    ww = paths.wakeword(cfg.name)
    features = [_feature(ww.positive_features, 2.0, True, "truncate_start")]
    if ww.real_positive_features.is_dir():
        features.append(
            _feature(
                ww.real_positive_features,
                cfg.real.positive_weight,
                True,
                "truncate_start",
            )
        )
    if ww.real_negative_features.is_dir():
        features.append(
            _feature(
                ww.real_negative_features, cfg.real.negative_weight, False, "random"
            )
        )
    for name, negative in NEGATIVE_SETS.items():
        features.append(
            _feature(
                paths.negatives / name,
                negative.sampling_weight,
                False,
                negative.truncation,
            )
        )

    return {
        "window_step_ms": FEATURE_STEP_MS,
        "train_dir": str(ww.train_dir.resolve()),
        "features": features,
        "training_steps": [cfg.train.steps],
        "positive_class_weight": [1],
        "negative_class_weight": [cfg.train.negative_class_weight],
        "learning_rates": [cfg.train.learning_rate],
        "batch_size": cfg.train.batch_size,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": cfg.train.eval_step_interval,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


def write(cfg: WakewordConfig, paths: Paths) -> Path:
    out = paths.wakeword(cfg.name).training_yaml
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(render(cfg, paths), f, sort_keys=False)
    return out
