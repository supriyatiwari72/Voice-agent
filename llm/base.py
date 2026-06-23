from abc import ABC, abstractmethod
from typing import Generator

class BaseLLM(ABC):
    """
    Abstract base class establishing the contract for Large Language Model (LLM) providers.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a complete text response for a given text prompt.

        Args:
            prompt (str): User transcript or orchestrated instruction.

        Returns:
            str: Full response text.
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream response tokens from the LLM.

        Args:
            prompt (str): User transcript or orchestrated instruction.

        Yields:
            str: Next token or phrase block.
        """
        pass
