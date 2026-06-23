import pytest
from stt.factory import STTFactory
from stt.base import BaseSTT

def test_stt_factory_valid_provider():
    """
    Test that STTFactory returns a valid BaseSTT for registered providers.
    """
    provider = STTFactory.get_provider("faster_whisper")
    assert isinstance(provider, BaseSTT)
    
    provider_dg = STTFactory.get_provider("deepgram")
    assert isinstance(provider_dg, BaseSTT)

def test_stt_factory_invalid_provider():
    """
    Test that STTFactory raises a ValueError for unregistered/unsupported providers.
    """
    with pytest.raises(ValueError) as excinfo:
        STTFactory.get_provider("invalid_provider")
    assert "Unsupported STT provider" in str(excinfo.value)

def test_stt_factory_empty_provider():
    """
    Test that STTFactory raises a ValueError when provider name is empty.
    """
    with pytest.raises(ValueError) as excinfo:
        STTFactory.get_provider("")
    assert "STT provider name must be specified" in str(excinfo.value)
