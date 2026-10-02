"""Filesystem layout: shared datasets, upstream checkouts, per-wakeword artifacts."""

import shutil
from dataclasses import dataclass
from pathlib import Path

# The project is installed editable, so this resolves to the repository.
EXTERNAL = Path(__file__).resolve().parents[2] / "external"


def piper_generator_dir() -> Path:
    return EXTERNAL / "piper-sample-generator"


@dataclass(frozen=True)
class Paths:
    data: Path
    output: Path

    @property
    def shared(self) -> Path:
        return self.data / "shared"

    @property
    def models(self) -> Path:
        return self.shared / "models"

    @property
    def rirs(self) -> Path:
        return self.shared / "augment" / "mit_rirs"

    @property
    def backgrounds(self) -> list[Path]:
        return [
            self.shared / "augment" / "fma_16k",
            self.shared / "augment" / "audioset_16k",
        ]

    @property
    def negatives(self) -> Path:
        return self.shared / "negative_datasets"

    def wakeword(self, name: str) -> "WakewordPaths":
        return WakewordPaths(self.data / name, self.output / name)


@dataclass(frozen=True)
class WakewordPaths:
    root: Path
    output: Path

    @property
    def positives(self) -> Path:
        return self.root / "positives"

    @property
    def real_positives(self) -> Path:
        return self.root / "real_positives"

    @property
    def real_negatives(self) -> Path:
        return self.root / "real_negatives"

    @property
    def positive_features(self) -> Path:
        return self.root / "features" / "positives"

    @property
    def real_positive_features(self) -> Path:
        return self.root / "features" / "real_positives"

    @property
    def real_negative_features(self) -> Path:
        return self.root / "features" / "real_negatives"

    @property
    def training_yaml(self) -> Path:
        return self.root / "training.yaml"

    @property
    def train_dir(self) -> Path:
        return self.root / "trained_model"

    @property
    def tflite(self) -> Path:
        return (
            self.train_dir
            / "tflite_stream_state_internal_quant"
            / "stream_state_internal_quant.tflite"
        )


def has_wavs(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.wav"))


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
