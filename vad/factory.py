from typing import Dict, Any, Type
from vad.base import BaseVAD
from vad.providers.dummy import DummyVAD
from vad.providers.silero import SileroVAD

class VADFactory:
    """
    Factory class responsible for validating and resolving BaseVAD instances.
    Supports DummyVAD and SileroVAD concrete implementations.
    """
    _providers: Dict[str, Type[BaseVAD]] = {
        "silero": SileroVAD,
        "webrtc": DummyVAD, # Placeholder for future WebRTC integration
        "dummy": DummyVAD
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseVAD:
        """
        Retrieves a VAD instance corresponding to the selected provider name.

        Args:
            name (str): The identifier of the provider (e.g. 'silero', 'dummy').
            config (Dict[str, Any]): Configuration settings map.

        Returns:
            BaseVAD: An instance of a Voice Activity Detection adapter.

        Raises:
            ValueError: If the provider is unsupported or unregistered.
        """
        if not name:
            raise ValueError("VAD provider name must be specified.")

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            raise ValueError(
                f"Unsupported VAD provider '{name}'. "
                f"Registered providers: {list(cls._providers.keys())}"
            )

        provider_cls = cls._providers[clean_name]
        return provider_cls(config or {})
