"""Per-wakeword configuration loaded from a YAML file."""

import dataclasses
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
PIPER_MODEL_NAME = "en_US-libritts_r-medium.pt"
# Spectrogram window step used for features, training and the ESPHome manifest.
FEATURE_STEP_MS = 10


class ConfigError(ValueError):
    pass


@dataclass
class PiperConfig:
    samples: int = 3000
    batch_size: int = 50
    model: str = PIPER_MODEL_NAME
    max_speakers: int | None = 800


@dataclass
class RealConfig:
    positive_weight: float = 3.0
    negative_weight: float = 1.0


@dataclass
class AugmentConfig:
    min_snr_db: int = -5
    max_snr_db: int = 10


@dataclass
class TrainConfig:
    steps: int = 10000
    batch_size: int = 128
    negative_class_weight: int = 20
    learning_rate: float = 0.001
    eval_step_interval: int = 500
    pointwise_filters: str = "64,64,64,64"
    stride: int = 3


@dataclass
class ManifestConfig:
    author: str = ""
    website: str = ""
    probability_cutoff: float = 0.97
    sliding_window_size: int = 5
    tensor_arena_size: int = 22860
    minimum_esphome_version: str = "2024.7"


@dataclass
class WakewordConfig:
    name: str
    phrase: str
    piper: PiperConfig = field(default_factory=PiperConfig)
    real: RealConfig = field(default_factory=RealConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        f = known[key]
        if is_dataclass(f.type):
            value = _build(f.type, value, f"{path}.{key}")  # type: ignore[arg-type]
        kwargs[key] = value
    missing = [
        f.name
        for f in fields(cls)
        if f.name not in kwargs
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ConfigError(f"{path}: missing required keys {missing}")
    return cls(**kwargs)


def load_config(path: Path) -> WakewordConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg: WakewordConfig = _build(WakewordConfig, data, path.name)
    if not isinstance(cfg.name, str) or not NAME_PATTERN.match(cfg.name):
        raise ConfigError(f"name must match {NAME_PATTERN.pattern}, got {cfg.name!r}")
    if not isinstance(cfg.phrase, str) or not cfg.phrase.strip():
        raise ConfigError("phrase must be a non-empty string")
    return cfg
