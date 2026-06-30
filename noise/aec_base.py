from abc import ABC, abstractmethod

class BaseEchoCanceller(ABC):
    """
    Abstract Base Class for Acoustic Echo Cancellation (AEC).
    """
    @abstractmethod
    def process(self, mic_chunk: bytes, speaker_reference: list) -> bytes:
        """
        Processes microphone audio bytes and filters out speaker echo.
        """
        pass
