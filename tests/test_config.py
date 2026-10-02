from pathlib import Path

import pytest
import yaml

from wakeword.config import ConfigError, load_config

EXAMPLE = Path(__file__).parents[1] / "wakewords" / "example.yaml"


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "ww.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_example_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.name == "example"
    assert cfg.phrase == "hey example"


def test_defaults_fill_missing_sections(tmp_path):
    cfg = load_config(write(tmp_path, {"name": "a", "phrase": "hey a"}))
    assert cfg.train.steps == 10000
    assert cfg.manifest.probability_cutoff == 0.97


@pytest.mark.parametrize(
    "data, match",
    [
        ({"name": "a", "phrase": "x", "train": {"stepz": 1}}, "stepz"),
        ({"name": "a", "phrase": "x", "train": 5}, "mapping"),
        ({"name": "Hey Word", "phrase": "x"}, "name"),
        ({"name": "a"}, "phrase"),
        ({"name": "a", "phrase": "   "}, "phrase"),
    ],
)
def test_invalid_config_rejected(tmp_path, data, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, data))
