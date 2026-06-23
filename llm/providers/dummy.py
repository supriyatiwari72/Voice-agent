from typing import Generator
from llm.base import BaseLLM

class DummyLLM(BaseLLM):
    """
    Concrete Dummy LLM that prints status and returns/streams hardcoded responses.
    """
    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, prompt: str) -> str:
        print("Dummy LLM:")
        print("Hello, how can I help you today?")
        return "Hello, how can I help you today?"

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        print("Dummy LLM (Streaming):")
        yield "Hello, "
        yield "how "
        yield "can "
        yield "I "
        yield "help "
        yield "you "
        yield "today?"
