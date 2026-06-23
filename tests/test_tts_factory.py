import pytest
from tts.factory import TTSFactory
from tts.base import BaseTTS

def test_tts_factory_valid_provider():
    """
    Test that TTSFactory returns a valid BaseTTS for registered providers.
    """
    provider = TTSFactory.get_provider("elevenlabs")
    assert isinstance(provider, BaseTTS)
    
    provider_cartesia = TTSFactory.get_provider("cartesia")
    assert isinstance(provider_cartesia, BaseTTS)

def test_tts_factory_invalid_provider():
    """
    Test that TTSFactory raises a ValueError for unregistered/unsupported providers.
    """
    with pytest.raises(ValueError) as excinfo:
        TTSFactory.get_provider("invalid_provider")
    assert "Unsupported TTS provider" in str(excinfo.value)

def test_tts_factory_empty_provider():
    """
    Test that TTSFactory raises a ValueError when provider name is empty.
    """
    with pytest.raises(ValueError) as excinfo:
        TTSFactory.get_provider("")
    assert "TTS provider name must be specified" in str(excinfo.value)
