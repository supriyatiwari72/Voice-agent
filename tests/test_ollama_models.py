"""
Tests for Ollama model variant factory keys.
Verifies that qwen2.5_3b and phi3_mini factory closures inject the correct model names,
and that generate_stream() works with mock fallback when Ollama is not running.
"""
import pytest
from llm.factory import LLMFactory
from llm.providers.ollama_streaming import OllamaStreamingLLM


BASE_CONFIG = {
    "models_meta": {
        "llm_providers": {
            "ollama": {
                "api_base": "http://localhost:11434",
                "model_name": "qwen2.5:3b",
                "temperature": 0.7,
                "max_tokens": 50,
                "timeout_seconds": 0.1,  # Force fast timeout for tests
                "max_retries": 1,
            }
        }
    }
}


class TestOllamaModelVariants:

    def test_factory_returns_ollama_instance_for_qwen_3b(self):
        provider = LLMFactory.get_provider("qwen2.5_3b", BASE_CONFIG)
        assert isinstance(provider, OllamaStreamingLLM)

    def test_factory_returns_ollama_instance_for_phi3_mini(self):
        provider = LLMFactory.get_provider("phi3_mini", BASE_CONFIG)
        assert isinstance(provider, OllamaStreamingLLM)

    def test_qwen_3b_has_correct_model_name(self):
        provider = LLMFactory.get_provider("qwen2.5_3b", BASE_CONFIG)
        assert provider.model == "qwen2.5:3b"

    def test_phi3_mini_has_correct_model_name(self):
        provider = LLMFactory.get_provider("phi3_mini", BASE_CONFIG)
        assert provider.model == "phi3:mini"

    def test_qwen_3b_mock_fallback_generates_tokens(self):
        """When Ollama is not running, falls back to mock tokens."""
        cfg = {
            "models_meta": {
                "llm_providers": {
                    "ollama": {
                        "api_base": "http://localhost:11434",
                        "model_name": "qwen2.5:3b",
                        "timeout_seconds": 0.1,
                        "max_retries": 1,
                    }
                }
            }
        }
        provider = LLMFactory.get_provider("qwen2.5_3b", cfg)
        tokens = list(provider.generate_stream("Hello"))
        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert len(full_text) > 0

    def test_phi3_mini_mock_fallback_generates_tokens(self):
        cfg = {
            "models_meta": {
                "llm_providers": {
                    "ollama": {
                        "api_base": "http://localhost:11434",
                        "model_name": "phi3:mini",
                        "timeout_seconds": 0.1,
                        "max_retries": 1,
                    }
                }
            }
        }
        provider = LLMFactory.get_provider("phi3_mini", cfg)
        tokens = list(provider.generate_stream("Hello"))
        assert len(tokens) > 0

    def test_generate_falls_back_to_string(self):
        cfg = {
            "models_meta": {
                "llm_providers": {
                    "ollama": {
                        "timeout_seconds": 0.1,
                        "max_retries": 1,
                    }
                }
            }
        }
        provider = LLMFactory.get_provider("qwen2.5_3b", cfg)
        result = provider.generate("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_system_prompt_is_used_when_set(self):
        cfg = {
            "system_prompt": "You are a helpful assistant.",
            "models_meta": {
                "llm_providers": {
                    "ollama": {
                        "timeout_seconds": 0.1,
                        "max_retries": 1,
                    }
                }
            }
        }
        provider = LLMFactory.get_provider("qwen2.5_3b", cfg)
        assert provider.system_prompt == "You are a helpful assistant."

    def test_messages_include_system_prompt(self):
        cfg = {
            "system_prompt": "Be concise.",
            "models_meta": {"llm_providers": {"ollama": {}}},
        }
        provider = LLMFactory.get_provider("qwen2.5_3b", cfg)
        messages = provider._build_messages("test prompt")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "test prompt"

    def test_messages_without_system_prompt(self):
        cfg = {"models_meta": {"llm_providers": {"ollama": {}}}}
        provider = LLMFactory.get_provider("qwen2.5_3b", cfg)
        messages = provider._build_messages("hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_models_are_independent_instances(self):
        """Verify closures produce independent instances, not shared state."""
        p1 = LLMFactory.get_provider("qwen2.5_3b", BASE_CONFIG)
        p2 = LLMFactory.get_provider("phi3_mini", BASE_CONFIG)
        assert p1 is not p2
        assert p1.model != p2.model

    def test_factory_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.get_provider("nonexistent_model_xyz", BASE_CONFIG)
