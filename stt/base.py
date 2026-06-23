from abc import ABC, abstractmethod

class BaseSTT(ABC):
    """
    Abstract base class establishing the contract for Speech-To-Text (STT) transcription providers.
    """

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe raw audio frame segments into a text string.

        Args:
            audio_data (bytes): Combined raw audio bytes representing a complete speech turn.

        Returns:
            str: Transcribed text output.
        """
        pass
