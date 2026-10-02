import pytest

from wakeword.data_setup import setup_piper_model


def test_unknown_piper_model_rejected(paths):
    with pytest.raises(ValueError, match="other.pt"):
        setup_piper_model(paths, "other.pt")
