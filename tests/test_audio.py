import numpy as np
import soundfile as sf

from wakeword.audio import is_normalized, normalize_dir


def write_stereo_44k(path):
    t = np.linspace(0, 0.5, 22050, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, np.stack([tone, tone], axis=1), 44100, subtype="FLOAT")


def test_normalize_dir_converts_to_16k_mono_int16(tmp_path):
    write_stereo_44k(tmp_path / "a.wav")
    assert not is_normalized(tmp_path / "a.wav")

    assert normalize_dir(tmp_path) == 1
    info = sf.info(tmp_path / "a.wav")
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert abs(info.frames - 8000) <= 1

    assert normalize_dir(tmp_path) == 0
