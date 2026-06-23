import pytest
from noise.factory import NoiseFactory
from noise.base import BaseNoiseCanceller

def test_noise_factory_valid_provider():
    """
    Test that NoiseFactory returns a valid BaseNoiseCanceller for registered providers.
    """
    provider = NoiseFactory.get_provider("rnnoise")
    assert isinstance(provider, BaseNoiseCanceller)
    
    provider_df = NoiseFactory.get_provider("deepfilternet")
    assert isinstance(provider_df, BaseNoiseCanceller)

def test_noise_factory_invalid_provider():
    """
    Test that NoiseFactory raises a ValueError for unregistered/unsupported providers.
    """
    with pytest.raises(ValueError) as excinfo:
        NoiseFactory.get_provider("invalid_provider")
    assert "Unsupported noise provider" in str(excinfo.value)

def test_noise_factory_empty_provider():
    """
    Test that NoiseFactory raises a ValueError when provider name is empty.
    """
    with pytest.raises(ValueError) as excinfo:
        NoiseFactory.get_provider("")
    assert "Noise provider name must be specified" in str(excinfo.value)
