from noise.aec_base import BaseEchoCanceller

class DummyEchoCanceller(BaseEchoCanceller):
    """
    Dummy Echo Canceller that bypasses processing and leaves audio unchanged.
    """
    def __init__(self, config: dict = None):
        pass

    def process(self, mic_chunk: bytes, speaker_reference: list) -> bytes:
        return mic_chunk
