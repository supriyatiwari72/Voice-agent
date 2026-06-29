import pytest
import numpy as np
from unittest.mock import patch
from noise.factory import NoiseFactory
from noise.providers.rnnoise import RNNoiseCanceller
from noise.base import BaseNoiseCanceller


@pytest.fixture
def mock_config():
    return {
        "models_meta": {
            "noise_providers": {
                "rnnoise": {
                    "attenuation": -15
                }
            }
        }
    }


def test_rnnoise_factory_creation(mock_config):
    canceller = NoiseFactory.get_provider("rnnoise", mock_config)
    assert isinstance(canceller, RNNoiseCanceller)
    assert isinstance(canceller, BaseNoiseCanceller)


def test_rnnoise_bypass_when_scipy_missing(mock_config):
    with patch("noise.providers.rnnoise.RNNoiseCanceller") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance._has_scipy = False
        mock_instance.process.side_effect = lambda x: x
        result = mock_instance.process(b"\x01\x02\x03\x04")
        assert result == b"\x01\x02\x03\x04"


def test_rnnoise_handles_empty_input(mock_config):
    canceller = RNNoiseCanceller(mock_config)
    assert canceller.process(b"") == b""


def test_rnnoise_preserves_length(mock_config):
    canceller = RNNoiseCanceller(mock_config)
    test_pcm = b"\x05\x00" * 160
    result = canceller.process(test_pcm)
    assert len(result) == len(test_pcm)
    audio_np = np.frombuffer(result, dtype=np.int16)
    assert len(audio_np) == 160


def test_rnnoise_actually_filters(mock_config):
    """With scipy available, the filter should modify the audio."""
    import os
    if os.path.basename(os.getcwd()) == "tests":
        os.chdir("..")
    canceller = RNNoiseCanceller(mock_config)
    if not canceller._has_scipy:
        pytest.skip("scipy not available")
    test_pcm = b"\x05\x00" * 800
    result = canceller.process(test_pcm)
    assert result != test_pcm
