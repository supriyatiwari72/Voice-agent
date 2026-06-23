from stt.base import BaseSTT

class DummySTT(BaseSTT):
    """
    Concrete Dummy STT that prints status and returns a hardcoded transcript.
    """
    def __init__(self, config=None):
        self.config = config or {}

    def transcribe(self, audio_data: bytes) -> str:
        print("Dummy STT:")
        print("Hello Voice Agent")
        return "Hello Voice Agent"
