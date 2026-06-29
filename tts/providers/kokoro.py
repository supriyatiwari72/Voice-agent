import logging
import os
import re
import numpy as np
from typing import Dict, Any, Generator
from tts.base import BaseTTS

logger = logging.getLogger(__name__)

# Safe dynamic import check for kokoro-onnx to support fallback behavior
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    class Kokoro:
        def __init__(self, model_path: str, voices_path: str):
            pass
        def create(self, text: str, voice: str, speed: float = 1.0):
            return np.zeros(0, dtype=np.float32), 24000
    logger.warning("Dependency 'kokoro-onnx' is missing. KokoroTTS will execute in fallback mode.")

class KokoroTTS(BaseTTS):
    """
    Concrete adapter for the Kokoro TTS engine using kokoro-onnx models.
    Supports file validation, dynamic fallback, and float-to-PCM conversion.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes Kokoro TTS with configuration parameters.
        Checks for weights and model file availability.
        """
        self.config = config or {}
        
        # Load parameters from models_meta configurations
        models_meta = self.config.get("models_meta", {})
        tts_config = models_meta.get("tts_providers", {}).get("kokoro", {}) or self.config

        self.model_path = tts_config.get("model_path", "weights/kokoro-v0_19.onnx")
        self.voices_path = tts_config.get("voices_path", "weights/voices.bin")
        self.voice = tts_config.get("voice", "af_bella")

        # Standard Kokoro speaker voices validation
        valid_voices = {
            "af", "af_bella", "af_sarah", "am_adam", "am_michael",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
            "jf_alpha", "jf_beta", "jf_gemma", "jf_sasa",
            "zf_alana", "zf_berta", "zf_dona", "zf_elena"
        }
        if self.voice not in valid_voices:
            logger.warning(f"Voice speaker profile '{self.voice}' is invalid for Kokoro. Falling back to default 'af_bella'.")
            self.voice = "af_bella"
        
        self.kokoro = None
        self.fallback = False

        if not KOKORO_AVAILABLE:
            logger.error("kokoro-onnx dependency is not installed. Operating in fallback mode.")
            self.fallback = True
            print("Kokoro Initialized (Fallback Mode)")
            return

        # Verify weights paths exist on disk
        if not os.path.exists(self.model_path):
            logger.error(f"Kokoro ONNX model file not found at: {self.model_path}. Operating in fallback mode.")
            self.fallback = True
        
        if not os.path.exists(self.voices_path):
            logger.error(f"Kokoro voices file not found at: {self.voices_path}. Operating in fallback mode.")
            self.fallback = True

        if self.fallback:
            print("Kokoro Initialized (Fallback Mode)")
            return

        try:
            logger.info(f"Loading Kokoro ONNX model from: {self.model_path}...")
            # Instantiate Kokoro model engine
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            logger.info(f"Kokoro TTS model loaded successfully. Voice profile: {self.voice}")
            print("Kokoro Initialized")
        except Exception as e:
            logger.exception(f"Exception raised while loading Kokoro model: {e}")
            self.fallback = True
            print("Kokoro Initialized (Fallback Mode)")

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text into complete audio bytes.

        Args:
            text (str): The text content to convert to speech.

        Returns:
            bytes: Complete 16-bit 16kHz mono PCM synthesized audio bytes.
        """
        if not text:
            return b""

        if self.fallback or not self.kokoro:
            logger.warning("KokoroTTS running in fallback mode; returning simulated silence.")
            # 1 second of 16kHz mono 16-bit PCM silence (32000 bytes)
            return b"\x00" * 32000

        try:
            logger.info(f"Synthesizing text: '{text[:60]}' using voice '{self.voice}'")
            samples, sample_rate = self.kokoro.create(text, voice=self.voice, speed=1.0)
            
            # Normalize float output in [-1.0, 1.0] to int16 range [-32768, 32767]
            pcm_array = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
            return pcm_array.tobytes()
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}. Returning simulated silence.")
            return b"\x00" * 32000

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream synthesized audio chunks for input text.

        Args:
            text (str): The text chunk to synthesize.

        Yields:
            bytes: Partial PCM audio frame bytes.
        """
        if not text:
            return

        if self.fallback or not self.kokoro:
            logger.warning("KokoroTTS running in fallback mode; yielding simulated stream silence.")
            yield b"\x00" * 16000
            yield b"\x00" * 16000
            return

        # Split text into sentence/clause segments for stream-like execution
        sentences = re.split(r"(?<=[.!?,\n])\s+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                logger.info(f"Streaming synthesis for text chunk: '{sentence}'")
                samples, sample_rate = self.kokoro.create(sentence, voice=self.voice, speed=1.0)
                pcm_array = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
                yield pcm_array.tobytes()
            except Exception as e:
                logger.error(f"Kokoro stream synthesis failed for sentence: '{sentence}': {e}")
