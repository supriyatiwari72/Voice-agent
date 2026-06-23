from noise.base import BaseNoiseCanceller

class DummyNoiseCanceller(BaseNoiseCanceller):
    """
    Concrete Dummy Noise Canceller that prints status once and returns the unmodified frame.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self._logged = False

    def process(self, audio_data: bytes) -> bytes:
        if not self._logged:
            print("Dummy Noise Reduction")
            self._logged = True
        return audio_data
