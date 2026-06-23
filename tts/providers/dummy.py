from typing import Generator
from tts.base import BaseTTS

class DummyTTS(BaseTTS):
    """
    Concrete Dummy TTS that prints status and returns/streams dummy audio bytes.
    """
    def __init__(self, config=None):
        self.config = config or {}

    def synthesize(self, text: str) -> bytes:
        print("Dummy TTS Generated Audio")
        return b"\x00" * 100

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        print("Dummy TTS Generated Audio (Streaming)")
        yield b"\x00" * 50
        yield b"\x00" * 50
