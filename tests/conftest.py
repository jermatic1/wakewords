from pathlib import Path

import pytest

from wakeword.config import WakewordConfig
from wakeword.paths import Paths


@pytest.fixture
def cfg() -> WakewordConfig:
    return WakewordConfig(name="demo", phrase="hey demo")


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(tmp_path / "data", tmp_path / "output")
