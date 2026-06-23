import pytest
from llm.factory import LLMFactory
from llm.base import BaseLLM

def test_llm_factory_valid_provider():
    """
    Test that LLMFactory returns a valid BaseLLM for registered providers.
    """
    provider = LLMFactory.get_provider("gemini")
    assert isinstance(provider, BaseLLM)
    
    provider_openai = LLMFactory.get_provider("openai")
    assert isinstance(provider_openai, BaseLLM)

def test_llm_factory_invalid_provider():
    """
    Test that LLMFactory raises a ValueError for unregistered/unsupported providers.
    """
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_provider("invalid_provider")
    assert "Unsupported LLM provider" in str(excinfo.value)

def test_llm_factory_empty_provider():
    """
    Test that LLMFactory raises a ValueError when provider name is empty.
    """
    with pytest.raises(ValueError) as excinfo:
        LLMFactory.get_provider("")
    assert "LLM provider name must be specified" in str(excinfo.value)
