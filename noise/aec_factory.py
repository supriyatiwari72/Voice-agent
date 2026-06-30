from typing import Dict, Any, Type
from noise.aec_base import BaseEchoCanceller
from noise.providers.dummy_aec import DummyEchoCanceller
from noise.providers.correlation_aec import CorrelationEchoCanceller

class AECFactory:
    """
    Factory class responsible for validating and resolving BaseEchoCanceller instances.
    """
    _providers: Dict[str, Type[BaseEchoCanceller]] = {
        "correlation": CorrelationEchoCanceller,
        "dummy": DummyEchoCanceller,
        "none": DummyEchoCanceller
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseEchoCanceller:
        """
        Retrieves an AEC instance corresponding to the selected provider name.
        """
        if not name:
            name = "dummy"

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            clean_name = "dummy"

        provider_cls = cls._providers[clean_name]
        return provider_cls(config or {})
