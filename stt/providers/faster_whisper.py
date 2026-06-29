import logging
import numpy as np
from typing import Dict, Any
from faster_whisper import WhisperModel
from stt.base import BaseSTT

logger = logging.getLogger(__name__)

class FasterWhisperSTT(BaseSTT):
    """
    Concrete adapter for the Faster Whisper Speech-To-Text engine.
    Normalizes input PCM bytes, performs transcription, and supports warm-up execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Faster Whisper model using configuration parameters.
        Runs a lightweight warm-up step.
        """
        self.config = config or {}
        
        # Load parameters from models_meta configurations
        models_meta = self.config.get("models_meta", {})
        stt_config = models_meta.get("stt_providers", {}).get("faster_whisper", {}) or self.config

        self.model_size = stt_config.get("model_size", "tiny")
        self.device = stt_config.get("device", "cpu")
        self.compute_type = stt_config.get("compute_type", "int8")
        self.beam_size = stt_config.get("beam_size", 5)

        logger.info(
            f"Loading Faster Whisper model: size={self.model_size}, "
            f"device={self.device}, compute_type={self.compute_type}..."
        )
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        print("Faster Whisper Initialized")

        # Perform one-time lightweight inference warm-up using 1 second of silence
        self._warm_up()

    def _warm_up(self) -> None:
        """
        Executes a dummy transcription to warm up the model and minimize subsequent latency.
        """
        try:
            warm_up_audio = np.zeros(16000, dtype=np.float32)
            segments, _ = self.model.transcribe(warm_up_audio, beam_size=self.beam_size, language="en")
            list(segments)
            logger.info("Faster Whisper model warm-up completed.")
        except Exception as e:
            logger.warning(f"Faster Whisper warm-up execution failed: {e}")

    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribes the raw 16-bit PCM bytes into text.

        Args:
            audio_data (bytes): Input PCM audio segment.

        Returns:
            str: Transcribed text, or empty string on failure.
        """
        if not audio_data:
            return ""

        try:
            # Convert 16-bit PCM bytes to normalized float32 numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_np) == 0:
                return ""

            # Call transcription — force English to avoid auto-detecting other languages
            segments, _ = self.model.transcribe(audio_np, beam_size=self.beam_size, language="en")
            
            # Exhaust segments generator to accumulate the final transcript string
            segment_texts = [segment.text for segment in segments]
            transcript = " ".join(segment_texts).strip()

            if transcript:
                print("Transcript:")
                print(transcript)

            return transcript

        except ValueError as e:
            logger.error(f"Failed to parse audio bytes: {e}")
            return ""
        except Exception as e:
            logger.exception(f"Faster Whisper transcription failed: {e}")
            return ""
