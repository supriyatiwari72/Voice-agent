from abc import ABC, abstractmethod

class BaseVAD(ABC):
    """
    Abstract base class establishing the contract for Voice Activity Detection (VAD) providers.
    """

    @abstractmethod
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Determine if the given audio chunk contains active speech.

        Args:
            audio_chunk (bytes): Raw audio frame bytes (e.g., 30ms of 16kHz PCM).

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        pass
