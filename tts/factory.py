from typing import Dict, Any, Type
from tts.base import BaseTTS
from tts.providers.dummy import DummyTTS

class TTSFactory:
    """
    Factory class responsible for validating and resolving BaseTTS instances.
    """
    _providers: Dict[str, Type[BaseTTS]] = {
        "elevenlabs": DummyTTS,
        "cartesia": DummyTTS,
        "piper": DummyTTS,
        "coqui": DummyTTS,
        "dummy": DummyTTS
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseTTS:
        """
        Retrieves a TTS instance corresponding to the selected provider name.

        Args:
            name (str): The identifier of the provider (e.g. 'elevenlabs').
            config (Dict[str, Any]): Configuration settings map.

        Returns:
            BaseTTS: An instance of a TTS adapter.

        Raises:
            ValueError: If the provider is unsupported or unregistered.
        """
        if not name:
            raise ValueError("TTS provider name must be specified.")

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            raise ValueError(
                f"Unsupported TTS provider '{name}'. "
                f"Registered providers: {list(cls._providers.keys())}"
            )

        provider_cls = cls._providers[clean_name]
        return provider_cls(config or {})
