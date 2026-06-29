import pytest
from llm.factory import LLMFactory
from llm.providers.ollama_streaming import OllamaStreamingLLM
from llm.providers.nvidia_nim_streaming import NVIDIANIMStreamingLLM
from llm.providers.vllm_streaming import VLLMStreamingLLM

@pytest.fixture
def base_config():
    return {
        "llm": {
            "ollama": {"url": "http://localhost:11434/api/generate"},
            "nvidia_nim": {"api_key": "MOCK_KEY"},
            "vllm": {"url": "http://localhost:8000/v1/chat/completions"}
        }
    }

@pytest.fixture(autouse=True)
def mock_network_calls(monkeypatch):
    import requests
    import urllib.request
    import urllib.error
    from unittest.mock import MagicMock
    
    # Mock requests
    monkeypatch.setattr(
        requests,
        "post",
        MagicMock(side_effect=requests.exceptions.ConnectionError("Mocked connection failure"))
    )
    
    # Mock urllib
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        MagicMock(side_effect=urllib.error.URLError("Mocked connection failure"))
    )

def test_llm_provider_instantiation(base_config):
    """
    Verify factories return the correct concrete streaming LLM adapters.
    """
    p1 = LLMFactory.get_provider("ollama_streaming", base_config)
    assert isinstance(p1, OllamaStreamingLLM)
    
    p2 = LLMFactory.get_provider("nvidia_nim_streaming", base_config)
    assert isinstance(p2, NVIDIANIMStreamingLLM)
    
    p3 = LLMFactory.get_provider("vllm_streaming", base_config)
    assert isinstance(p3, VLLMStreamingLLM)

def test_streaming_llm_generation(base_config):
    """
    Verify streaming LLM providers return a token generator in fallback mock mode.
    """
    providers = ["ollama_streaming", "nvidia_nim_streaming", "vllm_streaming"]
    
    for prov_name in providers:
        prov = LLMFactory.get_provider(prov_name, base_config)
        stream = prov.generate_stream("test prompt")
        
        tokens = list(stream)
        assert len(tokens) > 0
        joined = "".join(tokens)
        assert "streaming" in joined or "LLM" in joined or "Ollama" in joined or "NVIDIA" in joined or "vLLM" in joined

def test_streaming_llm_batch_fallback(base_config):
    """
    Verify that the generate() method acts as a batch fallback.
    """
    providers = ["ollama_streaming", "nvidia_nim_streaming", "vllm_streaming"]
    
    for prov_name in providers:
        prov = LLMFactory.get_provider(prov_name, base_config)
        text = prov.generate("hello")
        assert len(text) > 0
        assert "streaming" in text or "LLM" in text or "Ollama" in text or "NVIDIA" in text or "vLLM" in text
