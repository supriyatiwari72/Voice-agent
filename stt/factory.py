from typing import Dict, Any, Type
from stt.base import BaseSTT
from stt.providers.dummy import DummySTT
from stt.providers.faster_whisper import FasterWhisperSTT
from stt.providers.distil_whisper import DistilWhisperSTT
from stt.providers.deepgram_streaming import DeepgramStreamingSTT
from stt.providers.assemblyai_streaming import AssemblyAIStreamingSTT
from stt.providers.parakeet_streaming import ParakeetStreamingSTT
from stt.providers.moonshine import MoonshineSTT
from stt.providers.groq import GroqSTT

class STTFactory:
    """
    Factory class responsible for validating and resolving BaseSTT instances.
    """
    _providers: Dict[str, Type[BaseSTT]] = {
        "whisper": DummySTT,
        "faster_whisper": FasterWhisperSTT,
        "distil_whisper": DistilWhisperSTT,
        "moonshine": MoonshineSTT,
        "groq": GroqSTT,
        "deepgram": DummySTT,
        "deepgram_streaming": DeepgramStreamingSTT,
        "assemblyai_streaming": AssemblyAIStreamingSTT,
        "parakeet_streaming": ParakeetStreamingSTT,
        "google": DummySTT,
        "assemblyai": DummySTT,
        "dummy": DummySTT,
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseSTT:
        """
        Retrieves an STT instance corresponding to the selected provider name.

        Args:
            name (str): The identifier of the provider (e.g. 'faster_whisper').
            config (Dict[str, Any]): Configuration settings map.

        Returns:
            BaseSTT: An instance of a Speech-To-Text adapter.

        Raises:
            ValueError: If the provider is unsupported or unregistered.
        """
        if not name:
            raise ValueError("STT provider name must be specified.")

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            raise ValueError(
                f"Unsupported STT provider '{name}'. "
                f"Registered providers: {list(cls._providers.keys())}"
            )

        provider_cls = cls._providers[clean_name]
        return provider_cls(config or {})
