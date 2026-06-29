import logging
import os
import piper
from typing import Dict, Any, Generator

from tts.base import BaseTTS

logger = logging.getLogger(__name__)

class PiperStreamingTTS(BaseTTS):
    """
    Production Piper Streaming TTS Provider.
    Uses piper-tts Python library for synthesis.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        
        models_meta = self.config.get("models_meta", {})
        tts_config = models_meta.get("tts_providers", {}).get("piper", {}) or self.config.get("tts", {}).get("piper", {}) or {}
        
        self.model_path = os.path.abspath(tts_config.get("model_path") or "weights/en_US-lessac-medium.onnx")
        self.config_path = os.path.abspath(tts_config.get("config_path") or "weights/en_US-lessac-medium.onnx.json")
        
        self.voice = None
        self.fallback = True
        
        try:
            from piper.voice import PiperVoice
            self.PiperVoice = PiperVoice
        except ImportError:
            self.PiperVoice = None
            logger.warning("piper-tts not installed. Install with: pip install piper-tts")
        
        if self.PiperVoice and os.path.exists(self.model_path) and os.path.exists(self.config_path):
            try:
                espeak_dir = os.path.join(os.path.dirname(piper.__file__), "espeak-ng-data")
                self.voice = self.PiperVoice.load(
                    model_path=self.model_path,
                    config_path=self.config_path,
                    espeak_data_dir=espeak_dir,
                )
                self.fallback = False
                logger.info(f"PiperStreamingTTS initialized with model: {self.model_path} (sample_rate={self.voice.config.sample_rate})")
            except Exception as e:
                logger.error(f"Failed to load Piper voice: {e}")
        else:
            logger.warning("Piper model files not found or piper-tts not installed. PiperStreamingTTS will operate in mock fallback mode.")

    def _synthesize_chunk(self, text: str) -> bytes:
        if self.fallback or not self.voice:
            return b"\x00" * 32000
        
        try:
            chunks = list(self.voice.synthesize(text))
            if chunks:
                return chunks[0].audio_int16_bytes
            return b"\x00" * 32000
        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return b"\x00" * 32000

    def synthesize(self, text: str) -> bytes:
        return self._synthesize_chunk(text)

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        if self.fallback or not self.voice:
            logger.debug("PiperStreamingTTS mock streaming.")
            yield b"\x00" * 16000
            yield b"\x00" * 16000
            return

        try:
            for chunk in self.voice.synthesize(text):
                yield chunk.audio_int16_bytes
        except Exception as e:
            logger.error(f"Piper stream synthesis error: {e}")
            yield b"\x00" * 16000

    def stream_synthesize(self, text: str) -> Generator[bytes, None, None]:
        return self.synthesize_stream(text)
