import logging
import numpy as np
from typing import Dict, Any
from faster_whisper import WhisperModel
from stt.base import BaseSTT

logger = logging.getLogger(__name__)

class MoonshineSTT(BaseSTT):
    """
    Moonshine-inspired fast local STT using Faster-Whisper tiny model.
    Optimized for ultra-low latency: tiny model + beam_size=1 + no VAD filter.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        stt_config = models_meta.get("stt_providers", {}).get("moonshine", {}) or self.config

        self.model_size = stt_config.get("model_size", "tiny.en")
        self.device = stt_config.get("device", "cpu")
        self.compute_type = stt_config.get("compute_type", "int8")
        self.beam_size = stt_config.get("beam_size", 1)

        logger.info(
            f"Loading Moonshine (faster-whisper tiny) model: size={self.model_size}, "
            f"device={self.device}, compute_type={self.compute_type}..."
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

        self._warm_up()

    def _warm_up(self) -> None:
        try:
            warm_up_audio = np.zeros(16000, dtype=np.float32)
            segments, _ = self.model.transcribe(
                warm_up_audio,
                beam_size=self.beam_size,
                language="en",
                vad_filter=False,
            )
            list(segments)
            logger.info("Moonshine STT warm-up completed.")
        except Exception as e:
            logger.warning(f"Moonshine STT warm-up failed: {e}")

    def transcribe(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""

        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_np) == 0:
                return ""

            segments, _ = self.model.transcribe(
                audio_np,
                beam_size=self.beam_size,
                language="en",
                vad_filter=False,
            )

            transcript = " ".join(segment.text for segment in segments).strip()
            if transcript:
                print("Transcript:")
                print(transcript)

            return transcript

        except ValueError as e:
            logger.error(f"Failed to parse audio bytes: {e}")
            return ""
        except Exception as e:
            logger.exception(f"Moonshine STT transcription failed: {e}")
            return ""
