from vad.base import BaseVAD

class DummyVAD(BaseVAD):
    """
    Concrete Dummy VAD that prints status and returns speech status toggling.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self._triggered = False

    def is_speech(self, audio_chunk: bytes) -> bool:
        if not self._triggered:
            print("Dummy VAD Detected Speech")
            self._triggered = True
            return True
        return False
