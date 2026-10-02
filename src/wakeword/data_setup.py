"""Downloads the shared augmentation and negative datasets on first use."""

import shutil
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm

from wakeword.audio import normalize_wav, write_wav
from wakeword.config import PIPER_MODEL_NAME
from wakeword.download import download_hf_file, download_url
from wakeword.paths import Paths, piper_generator_dir, remove
from wakeword.train_config import NEGATIVE_SETS

PIPER_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
    + PIPER_MODEL_NAME
)
NEGATIVES_REPO = "kahrendt/microwakeword"


def _done(directory: Path) -> Path:
    return directory / ".done"


def _convert_all(sources: Iterable[Path], out_dir: Path, desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in tqdm(list(sources), desc=desc):
        normalize_wav(src, out_dir / f"{src.stem}.wav")


def setup_piper_model(paths: Paths, model_name: str) -> Path:
    """Downloads the generator weights and places its JSON config beside them."""
    if model_name != PIPER_MODEL_NAME:
        raise ValueError(f"no download source known for piper model {model_name!r}")
    model = download_url(PIPER_MODEL_URL, paths.models / model_name)
    config = model.with_name(model_name + ".json")
    if not config.exists():
        source = piper_generator_dir() / "models" / config.name
        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found; run `git submodule update --init`"
            )
        shutil.copy(source, config)
    return model


def setup_rirs(paths: Paths) -> None:
    out_dir = paths.rirs
    if _done(out_dir).exists():
        return
    remove(out_dir)
    out_dir.mkdir(parents=True)
    rows = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    for row in tqdm(rows, desc="mit_rirs"):
        name = Path(row["audio"]["path"]).name
        samples = (np.asarray(row["audio"]["array"]) * 32767).astype(np.int16)
        write_wav(out_dir / name, samples)
    _done(out_dir).touch()


def _setup_background(
    paths: Paths,
    name: str,
    repo_id: str,
    filename: str,
    extract: Callable[[Path, Path], None],
    pattern: str,
) -> None:
    out_dir = paths.shared / "augment" / name
    if _done(out_dir).exists():
        return
    raw = paths.shared / "raw" / name
    archive = download_hf_file(repo_id, filename, raw)
    extract(archive, raw)
    remove(out_dir)
    _convert_all(raw.rglob(pattern), out_dir, name)
    remove(raw)
    _done(out_dir).touch()


def _untar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def _unzip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def setup_audioset(paths: Paths) -> None:
    _setup_background(
        paths,
        "audioset_16k",
        "agkphysics/AudioSet",
        "data/bal_train09.tar",
        _untar,
        "*.flac",
    )


def setup_fma(paths: Paths) -> None:
    _setup_background(
        paths, "fma_16k", "mchl914/fma_xsmall", "fma_xs.zip", _unzip, "*.mp3"
    )


def setup_negatives(paths: Paths) -> None:
    out_dir = paths.negatives
    if _done(out_dir).exists():
        return
    raw = paths.shared / "raw" / "negatives"
    for name in NEGATIVE_SETS:
        zip_path = download_hf_file(NEGATIVES_REPO, f"{name}.zip", raw)
        remove(out_dir / name)
        _unzip(zip_path, out_dir)
    remove(raw)
    _done(out_dir).touch()
