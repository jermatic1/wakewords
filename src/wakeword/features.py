"""Turns audio clips into augmented spectrogram feature sets for training."""

import logging
from pathlib import Path
from typing import NamedTuple

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

from wakeword import data_setup
from wakeword.config import FEATURE_STEP_MS, WakewordConfig
from wakeword.paths import Paths, has_wavs, remove

# Below this the train/validation/test split leaves sets too small to be useful.
MIN_CLIPS_FOR_SPLIT = 10
NEGATIVE_CHUNK_S = 10.0
AUGMENTATION_PROBABILITIES = {
    "SevenBandParametricEQ": 0.1,
    "TanhDistortion": 0.1,
    "PitchShift": 0.1,
    "BandStopFilter": 0.1,
    "AddColorNoise": 0.1,
    "AddBackgroundNoise": 0.75,
    "Gain": 1.0,
    "RIR": 0.5,
}
log = logging.getLogger(__name__)


class Split(NamedTuple):
    set_name: str
    clip_split: str | None
    repeat: int


def build_augmenter(cfg: WakewordConfig, paths: Paths) -> Augmentation:
    backgrounds = [str(p) for p in paths.backgrounds if has_wavs(p)]
    if not backgrounds:
        log.warning("no background noise sets found; augmenting with RIR only")
    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities=AUGMENTATION_PROBABILITIES,
        impulse_paths=[str(paths.rirs)] if has_wavs(paths.rirs) else [],
        background_paths=backgrounds,
        background_min_snr_db=cfg.augment.min_snr_db,
        background_max_snr_db=cfg.augment.max_snr_db,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def _load_clips(directory: Path) -> tuple[Clips, list[Split]]:
    count = len(list(directory.glob("*.wav")))
    split = count >= MIN_CLIPS_FOR_SPLIT
    clips = Clips(
        input_directory=str(directory),
        file_pattern="*.wav",
        random_split_seed=10 if split else None,
        split_count=0.1,
    )
    if not split:
        return clips, [Split("training", None, 1)]
    return clips, [
        Split("training", "train", 2),
        Split("validation", "validation", 1),
        Split("testing", "test", 1),
    ]


def _write_mmap(out_dir: Path, generator) -> None:
    remove(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=str(out_dir),
        sample_generator=generator,
        batch_size=100,
        verbose=True,
    )


def generate_positive_features(
    clips_dir: Path, out_dir: Path, augmenter: Augmentation
) -> None:
    clips, splits = _load_clips(clips_dir)
    for split in splits:
        # Sliding the spectrogram simulates streaming inference; the test set
        # runs the real streaming model so it does not need it.
        slide_frames = 1 if split.set_name == "testing" else 10
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=FEATURE_STEP_MS,
        )
        _write_mmap(
            out_dir / split.set_name / "wakeword_mmap",
            spectrograms.spectrogram_generator(
                split=split.clip_split, repeat=split.repeat
            ),
        )


def generate_negative_features(clips_dir: Path, out_dir: Path) -> None:
    """Long ambient recordings are cut into fixed chunks and used unaugmented."""
    clips, splits = _load_clips(clips_dir)
    for split in splits:
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=None,
            step_ms=FEATURE_STEP_MS,
            split_spectrogram_duration_s=NEGATIVE_CHUNK_S,
        )
        _write_mmap(
            out_dir / split.set_name / "negative_mmap",
            spectrograms.spectrogram_generator(split=split.clip_split),
        )


def build_features(
    cfg: WakewordConfig, paths: Paths, skip_background: bool = False
) -> None:
    ww = paths.wakeword(cfg.name)
    if not has_wavs(ww.positives):
        raise FileNotFoundError(f"no positive samples in {ww.positives}; run generate")
    data_setup.setup_rirs(paths)
    if not skip_background:
        data_setup.setup_audioset(paths)
        data_setup.setup_fma(paths)
    augmenter = build_augmenter(cfg, paths)

    log.info("featurizing synthetic positives")
    generate_positive_features(ww.positives, ww.positive_features, augmenter)

    if has_wavs(ww.real_positives):
        log.info("featurizing real positives")
        generate_positive_features(
            ww.real_positives, ww.real_positive_features, augmenter
        )
    else:
        remove(ww.real_positive_features)

    if has_wavs(ww.real_negatives):
        log.info("featurizing real negatives")
        generate_negative_features(ww.real_negatives, ww.real_negative_features)
    else:
        remove(ww.real_negative_features)
