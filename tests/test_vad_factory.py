import pytest
from vad.factory import VADFactory
from vad.base import BaseVAD

def test_vad_factory_valid_provider():
    """
    Test that VADFactory returns a valid BaseVAD for registered providers.
    """
    provider = VADFactory.get_provider("silero")
    assert isinstance(provider, BaseVAD)
    
    provider_webrtc = VADFactory.get_provider("webrtc")
    assert isinstance(provider_webrtc, BaseVAD)

def test_vad_factory_invalid_provider():
    """
    Test that VADFactory raises a ValueError for unregistered/unsupported providers.
    """
    with pytest.raises(ValueError) as excinfo:
        VADFactory.get_provider("invalid_provider")
    assert "Unsupported VAD provider" in str(excinfo.value)

def test_vad_factory_empty_provider():
    """
    Test that VADFactory raises a ValueError when provider name is empty.
    """
    with pytest.raises(ValueError) as excinfo:
        VADFactory.get_provider("")
    assert "VAD provider name must be specified" in str(excinfo.value)
