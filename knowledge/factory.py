from typing import Dict, Any, Type
from knowledge.base import BaseKnowledgeRetriever
from knowledge.providers.json_retriever import JSONKnowledgeRetriever


class KnowledgeFactory:
    """
    Factory class responsible for validating and resolving BaseKnowledgeRetriever instances.
    """
    _providers: Dict[str, Type[BaseKnowledgeRetriever]] = {
        "json": JSONKnowledgeRetriever,
    }

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any] = None) -> BaseKnowledgeRetriever:
        """
        Retrieves a KnowledgeRetriever instance corresponding to the selected provider name.

        Args:
            name (str): The identifier of the provider (e.g. 'json').
            config (Dict[str, Any]): Configuration settings map.

        Returns:
            BaseKnowledgeRetriever: An instance of a knowledge retrieval adapter.

        Raises:
            ValueError: If the provider is unsupported or unregistered.
        """
        if not name:
            raise ValueError("Knowledge provider name must be specified.")

        clean_name = name.strip().lower()
        if clean_name not in cls._providers:
            raise ValueError(
                f"Unsupported knowledge provider '{name}'. "
                f"Registered providers: {list(cls._providers.keys())}"
            )

        provider_cls = cls._providers[clean_name]
        resolved_config = dict(config) if config else {}
        resolved_config["_knowledge_provider_name"] = clean_name
        return provider_cls(resolved_config)
