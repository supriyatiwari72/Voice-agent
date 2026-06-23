from typing import Dict, Any, Type
from noise.base import BaseNoiseCanceller
from noise.providers.dummy import DummyNoiseCanceller

class NoiseFactory:
    """
    Factory class responsible for validating and resolving BaseNoiseCanceller instances.
    """
    _providers: Dict[str, Type[BaseNoiseCanceller]] = {
        "rnnoise": DummyNoiseCanceller,
        "deepfilternet": DummyNoiseCanceller,
        "dummy": DummyNoiseCanceller
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseNoiseCanceller:
        """
        Retrieves a noise canceller instance corresponding to the selected provider name.

        Args:
            name (str): The identifier of the provider (e.g. 'rnnoise').
            config (Dict[str, Any]): Configuration settings map.

        Returns:
            BaseNoiseCanceller: An instance of a noise cancellation adapter.

        Raises:
            ValueError: If the provider is unsupported or unregistered.
        """
        if not name:
            raise ValueError("Noise provider name must be specified.")

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            raise ValueError(
                f"Unsupported noise provider '{name}'. "
                f"Registered providers: {list(cls._providers.keys())}"
            )

        provider_cls = cls._providers[clean_name]
        return provider_cls(config or {})
