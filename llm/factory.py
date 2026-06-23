from typing import Dict, Any, Type
from llm.base import BaseLLM
from llm.providers.dummy import DummyLLM

class LLMFactory:
    """
    Factory class responsible for validating and resolving BaseLLM instances.
    """
    _providers: Dict[str, Type[BaseLLM]] = {
        "openai": DummyLLM,
        "gemini": DummyLLM,
        "claude": DummyLLM,
        "llama": DummyLLM,
        "deepseek": DummyLLM,
        "dummy": DummyLLM
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseLLM:
        """
        Retrieves an LLM instance corresponding to the selected provider name.

        Args:
            name (str): The identifier of the provider (e.g. 'gemini').
            config (Dict[str, Any]): Configuration settings map.

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

        provider_cls = cls._providers[clean_name]
        return provider_cls(config or {})
