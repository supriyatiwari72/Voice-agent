from abc import ABC, abstractmethod

class BaseNoiseCanceller(ABC):
    """
    Abstract base class establishing the contract for noise cancellation providers.
    Every noise cancellation provider must implement this interface.
    """

    @abstractmethod
    def process(self, audio_data: bytes) -> bytes:
        """
        Process a chunk of raw audio data to filter out background noise.

        Args:
            audio_data (bytes): The raw incoming audio frames (typically PCM).

        Returns:
            bytes: The noise-attenuated audio frames.
        """
        pass
