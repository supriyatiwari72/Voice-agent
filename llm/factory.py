import copy
from typing import Dict, Any, Type, Callable, Union
from llm.base import BaseLLM
from llm.providers.dummy import DummyLLM
from llm.providers.open_source_llm import OpenSourceLLM
from llm.providers.ollama_streaming import OllamaStreamingLLM
from llm.providers.nvidia_nim_streaming import NVIDIANIMStreamingLLM
from llm.providers.vllm_streaming import VLLMStreamingLLM
from llm.providers.groq_llm import GroqLLM
from llm.providers.openrouter_llm import OpenRouterLLM


def _make_ollama(model_name: str) -> Callable[[Dict[str, Any]], OllamaStreamingLLM]:
    """
    Factory closure that produces an OllamaStreamingLLM configured for a specific model.
    Injects the model name into models_meta so the provider reads it correctly.
    """
    def factory(config: Dict[str, Any]) -> OllamaStreamingLLM:
        cfg = copy.deepcopy(config) if config else {}
        cfg.setdefault("models_meta", {})
        cfg["models_meta"].setdefault("llm_providers", {})

        # Merge existing ollama config (for api_base, timeout etc.) with the target model
        existing = cfg["models_meta"]["llm_providers"].get("ollama", {})
        cfg["models_meta"]["llm_providers"]["ollama"] = {
            **existing,
            "model_name": model_name,
        }
        return OllamaStreamingLLM(cfg)
    return factory


class LLMFactory:
    """
    Factory class responsible for validating and resolving BaseLLM instances.

    Registered provider keys:
    ─── Local Ollama Models (require: ollama pull <model>) ─────────────────────
      qwen2.5_1.5b    → qwen2.5:1.5b
      qwen2.5_3b      → qwen2.5:3b    ← Demo default
      phi3_mini       → phi3:mini
    ─── API-Based Providers ────────────────────────────────────────────────────
      groq            → Groq API (llama3-8b-8192 default) — set GROQ_API_KEY
    ─── OpenAI-Compatible Endpoints ────────────────────────────────────────────
      ollama_streaming → Generic Ollama (model from models.yaml)
      nvidia_nim_streaming → NVIDIA NIM
      vllm_streaming   → vLLM local server
    ─── General / Legacy ────────────────────────────────────────────────────────
      qwen, ollama    → OpenSourceLLM via Ollama endpoint
      dummy           → Mock LLM (for testing)
    """

    # Callable entries in _providers can be either a class (Type[BaseLLM])
    # or a factory function (config → BaseLLM).
    _providers: Dict[str, Union[Type[BaseLLM], Callable[[Dict[str, Any]], BaseLLM]]] = {
        # ── Ollama model variants (factory closures) ──────────────────────────
        "qwen2.5_1.5b": _make_ollama("qwen2.5:1.5b"),
        "qwen2.5_3b":   _make_ollama("qwen2.5:3b"),
        "phi3_mini":    _make_ollama("phi3:mini"),

        # ── API-based providers ───────────────────────────────────────────────
        "groq": GroqLLM,
        "openrouter": OpenRouterLLM,

        # ── Generic Ollama / OpenAI-compatible ────────────────────────────────
        "ollama_streaming":      OllamaStreamingLLM,
        "nvidia_nim_streaming":  NVIDIANIMStreamingLLM,
        "vllm_streaming":        VLLMStreamingLLM,

        # ── Legacy keys (backward compatible) ────────────────────────────────
        "openai":   DummyLLM,
        "gemini":   DummyLLM,
        "claude":   DummyLLM,
        "llama":    DummyLLM,
        "deepseek": DummyLLM,
        "qwen":     OpenSourceLLM,
        "ollama":   OpenSourceLLM,
        "dummy":    DummyLLM,
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseLLM:
        """
        Retrieves an LLM instance corresponding to the selected provider name.

        Args:
            name (str): Provider key (e.g. 'qwen2.5_3b', 'groq', 'dummy').
            config (Dict[str, Any]): Full config dict (including models_meta).

        Returns:
            BaseLLM: An instance of an LLM adapter.

        Raises:
            ValueError: If the provider is unsupported or unregistered.
        """
        if not name:
            raise ValueError("LLM provider name must be specified.")

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            raise ValueError(
                f"Unsupported LLM provider '{name}'. "
                f"Registered providers: {list(cls._providers.keys())}"
            )

        entry = cls._providers[clean_name]
        cfg = config or {}

        # If entry is a factory closure (callable but not a class), call it directly
        if callable(entry) and not isinstance(entry, type):
            return entry(cfg)

        # Otherwise it's a class — instantiate normally
        return entry(cfg)
