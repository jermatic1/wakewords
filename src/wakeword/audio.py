"""Audio normalization to the 16 kHz mono int16 format training expects."""

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
log = logging.getLogger(__name__)


def is_normalized(path: Path) -> bool:
    info = sf.info(path)
    return (
        info.samplerate == SAMPLE_RATE
        and info.channels == 1
        and info.subtype == "PCM_16"
    )


def load_normalized(path: Path) -> np.ndarray:
    """Loads any audio file as 16 kHz mono int16 samples."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def normalize_wav(src: Path, dst: Path) -> None:
    write_wav(dst, load_normalized(src))


def normalize_dir(directory: Path) -> int:
    """Rewrites every WAV in the directory in place if it is not already normalized."""
    count = 0
    for path in sorted(directory.glob("*.wav")):
        if is_normalized(path):
            continue
        normalize_wav(path, path)
        count += 1
    log.info("normalized %d files in %s", count, directory)
    return count
