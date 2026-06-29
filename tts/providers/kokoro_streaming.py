import logging
from typing import Dict, Any, Generator
from tts.base import BaseTTS
from tts.providers.kokoro import KokoroTTS

logger = logging.getLogger(__name__)

class KokoroStreamingTTS(KokoroTTS):
    """
    Production Kokoro Streaming TTS Provider.
    Inherits from the verified KokoroTTS engine and maps stream_synthesize to synthesise_stream.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def stream_synthesize(self, text: str) -> Generator[bytes, None, None]:
        """
        Interface method returning a generator yielding partial synthesized audio chunk PCM bytes.
        """
        logger.info(f"KokoroStreamingTTS: Initiating streaming synthesis for '{text[:40]}...'")
        return self.synthesize_stream(text)
