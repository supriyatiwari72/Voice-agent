import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from vad.factory import VADFactory
from vad.providers.silero import SileroVAD
from vad.base import BaseVAD

@pytest.fixture
def base_config():
    return {
        "audio": {
            "sample_rate": 16000
        },
        "vad_providers": {
            "silero": {
                "threshold": 0.5
            }
        }
    }

def test_silero_model_loading(base_config):
    """
    Test 1: Model loads successfully on initialization.
    """
    vad = SileroVAD(base_config)
    assert vad.model is not None
    assert vad.threshold == 0.5

def test_silero_returns_boolean(base_config):
    """
    Test 2: is_speech() returns a boolean type.
    """
    vad = SileroVAD(base_config)
    # Generate 512 samples of quiet/silent PCM audio data (1024 bytes)
    silent_pcm = b"\x00" * 1024
    result = vad.is_speech(silent_pcm)
    assert isinstance(result, bool)

def test_silero_handles_empty_audio(base_config):
    """
    Test 3: is_speech() returns False and handles empty bytes cleanly.
    """
    vad = SileroVAD(base_config)
    assert vad.is_speech(b"") is False
    assert vad.is_speech(None) is False

def test_silero_handles_invalid_audio_length(base_config):
    """
    Test 4: is_speech() handles invalid length and very short inputs safely.
    """
    vad = SileroVAD(base_config)
    # Odd byte length (1 byte)
    assert vad.is_speech(b"\x00") is False
    # Very short length
    assert vad.is_speech(b"\x00" * 4) is False

def test_silero_factory_creation(base_config):
    """
    Test 5: Factory resolves and instantiates SileroVAD correctly.
    """
    provider = VADFactory.get_provider("silero", base_config)
    assert isinstance(provider, SileroVAD)
    assert isinstance(provider, BaseVAD)

def test_silero_threshold_gating(base_config):
    """
    Test 6: Threshold behavior gates output correctly depending on speech probability.
    """
    vad = SileroVAD(base_config)
    
    # Mock the internal model to return specific speech probabilities
    mock_model = MagicMock()
    vad.model = mock_model

    # Sub-threshold probability
    mock_model.return_value = torch.tensor([[0.4]])
    pcm_chunk = b"\x00" * 1024
    assert vad.is_speech(pcm_chunk) is False

    # Above-threshold probability
    mock_model.return_value = torch.tensor([[0.8]])
    assert vad.is_speech(pcm_chunk) is True

def test_silero_provider_switch_validation(base_config):
    """
    Test 7: Factory resolves both dummy and silero providers correctly.
    """
    dummy_provider = VADFactory.get_provider("dummy", base_config)
    assert dummy_provider is not None
    
    silero_provider = VADFactory.get_provider("silero", base_config)
    assert silero_provider is not None
