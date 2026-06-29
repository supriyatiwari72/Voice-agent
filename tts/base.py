from abc import ABC, abstractmethod
from typing import Generator

class BaseTTS(ABC):
    """
    Abstract base class establishing the contract for Text-To-Speech (TTS) synthesis providers.
    """

    def supports_streaming_tts(self) -> bool:
        """Returns True if this provider can stream synthesis audio chunks."""
        return hasattr(self, "stream_synthesize")

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text into complete audio bytes.

        Args:
            text (str): The text content to convert to speech.

        Returns:
            bytes: Complete synthesized audio waveform bytes (e.g. WAV/PCM).
        """
        pass

    @abstractmethod
    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream synthesized audio chunks for input text.

        Args:
            text (str): The text chunk/tokens to synthesize in real time.

        Yields:
            bytes: Partial audio frame bytes.
        """
        pass
