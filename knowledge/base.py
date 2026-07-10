from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseKnowledgeRetriever(ABC):
    """
    Abstract base class establishing the contract for Knowledge Retrieval providers.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge entries relevant to the given query.

        Args:
            query (str): The user's search query or utterance.

        Returns:
            List[Dict[str, Any]]: A list of matching knowledge entries ranked by relevance,
                                  or an empty list if no matches are found.
        """
        pass
