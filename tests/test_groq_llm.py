"""
Tests for GroqLLM provider.
Verifies instantiation, API key handling, mock fallback, and correct API base URL.
"""
import pytest
from llm.factory import LLMFactory
from llm.providers.groq_llm import GroqLLM


BASE_CONFIG = {
    "models_meta": {
        "llm_providers": {
            "groq": {
                "api_base": "https://api.groq.com/openai/v1",
                "model_name": "llama3-8b-8192",
                "temperature": 0.7,
                "max_tokens": 50,
                "timeout_seconds": 0.1,
                "max_retries": 1,
            }
        }
    }
}

NO_KEY_CONFIG = {
    "models_meta": {
        "llm_providers": {
            "groq": {
                "api_key": "",  # No key
                "timeout_seconds": 0.1,
                "max_retries": 1,
            }
        }
    }
}


class TestGroqLLM:

    def test_factory_returns_groq_instance(self):
        provider = LLMFactory.get_provider("groq", BASE_CONFIG)
        assert isinstance(provider, GroqLLM)

    def test_groq_uses_correct_api_base(self):
        provider = LLMFactory.get_provider("groq", BASE_CONFIG)
        assert "groq.com" in provider.url

    def test_groq_mock_fallback_when_no_api_key(self):
        """Without an API key, GroqLLM returns mock tokens rather than crashing."""
        provider = LLMFactory.get_provider("groq", NO_KEY_CONFIG)
        tokens = list(provider.generate_stream("Hello"))
        assert len(tokens) > 0
        full_text = "".join(tokens)
        # Mock response should mention the key is missing
        assert len(full_text) > 0

    def test_groq_generate_returns_string_without_key(self):
        provider = LLMFactory.get_provider("groq", NO_KEY_CONFIG)
        result = provider.generate("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_groq_empty_prompt_returns_empty(self):
        provider = LLMFactory.get_provider("groq", NO_KEY_CONFIG)
        result = provider.generate("")
        assert result == ""

    def test_groq_stream_empty_prompt_returns_nothing(self):
        provider = LLMFactory.get_provider("groq", NO_KEY_CONFIG)
        tokens = list(provider.generate_stream(""))
        assert tokens == []

    def test_groq_api_key_from_config(self):
        cfg = {
            "models_meta": {
                "llm_providers": {
                    "groq": {
                        "api_key": "gsk_test_12345",
                        "timeout_seconds": 0.1,
                        "max_retries": 1,
                    }
                }
            }
        }
        provider = LLMFactory.get_provider("groq", cfg)
        # Key is stored internally
        assert provider._groq_api_key == "gsk_test_12345"

    def test_groq_with_api_key_attempts_real_call_but_fails_gracefully(self):
        """With a fake API key, the real HTTP call should fail gracefully (not raise)."""
        cfg = {
            "models_meta": {
                "llm_providers": {
                    "groq": {
                        "api_key": "gsk_fake_key_for_testing",
                        "api_base": "https://api.groq.com/openai/v1",
                        "model_name": "llama3-8b-8192",
                        "timeout_seconds": 2.0,
                        "max_retries": 1,
                    }
                }
            }
        }
        provider = LLMFactory.get_provider("groq", cfg)
        # Should either return tokens (if reachable) or an error string — not raise
        try:
            tokens = list(provider.generate_stream("hello"))
            assert isinstance(tokens, list)
        except Exception:
            pytest.fail("GroqLLM.generate_stream() raised an exception instead of failing gracefully")

    def test_groq_is_registered_in_factory(self):
        assert "groq" in LLMFactory._providers

    def test_groq_invalid_key_env_fallback(self, monkeypatch):
        """Test env var resolution for GROQ_API_KEY."""
        monkeypatch.setenv("GROQ_API_KEY", "env_test_key_abc")
        provider = GroqLLM({})
        assert provider._groq_api_key == "env_test_key_abc"
