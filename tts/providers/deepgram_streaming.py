import logging
import os
import time
from typing import Dict, Any, Generator
from tts.base import BaseTTS

logger = logging.getLogger(__name__)

class DeepgramStreamingTTS(BaseTTS):
    """
    Deepgram TTS Provider using deepgram-sdk.
    Supports streaming synthesis for low-latency voice output.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        tts_config = (
            models_meta.get("tts_providers", {}).get("deepgram", {})
            or self.config.get("tts", {}).get("deepgram", {})
            or {}
        )

        self.api_key = tts_config.get("api_key") or os.environ.get("DEEPGRAM_API_KEY", "")
        self.model = tts_config.get("model", "aura-asteria-en")
        self.voice = tts_config.get("voice", "aura-asteria-en")
        self.sample_rate = int(tts_config.get("sample_rate", 24000))
        self.encoding = tts_config.get("encoding", "linear16")
        self.container = tts_config.get("container", "wav")

        self.fallback = not bool(self.api_key)
        if self.fallback:
            logger.warning("DeepgramStreamingTTS: No DEEPGRAM_API_KEY found. Falling back to silent mock.")

    def synthesize(self, text: str) -> bytes:
        if self.fallback:
            return b"\x00" * 32000

        try:
            from deepgram import DeepgramClient, SpeakOptions
            deepgram = DeepgramClient(self.api_key)
            options = SpeakOptions(
                model=self.model,
                voice=self.voice,
                encoding=self.encoding,
                sample_rate=self.sample_rate,
                container=self.container,
            )
            response = deepgram.speak.v("1").stream_raw({"text": text}, options)
            return response.read()
        except Exception as e:
            logger.error(f"Deepgram TTS synthesis error: {e}")
            return b"\x00" * 32000

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        if self.fallback:
            yield b"\x00" * 16000
            yield b"\x00" * 16000
            return

        try:
            from deepgram import DeepgramClient, SpeakOptions
            deepgram = DeepgramClient(self.api_key)
            options = SpeakOptions(
                model=self.model,
                voice=self.voice,
                encoding=self.encoding,
                sample_rate=self.sample_rate,
                container=self.container,
            )
            with deepgram.speak.v("1").stream_raw({"text": text}, options) as response:
                for chunk in response:
                    yield chunk
        except Exception as e:
            logger.error(f"Deepgram TTS stream synthesis error: {e}")
            yield b"\x00" * 16000

    def stream_synthesize(self, text: str) -> Generator[bytes, None, None]:
        return self.synthesize_stream(text)
