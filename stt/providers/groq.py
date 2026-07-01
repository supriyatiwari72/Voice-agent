import io
import os
import wave
import logging
import requests
from typing import Dict, Any
from stt.base import BaseSTT

logger = logging.getLogger(__name__)

class GroqSTT(BaseSTT):
    """
    Concrete adapter for the Groq Cloud Speech-To-Text API.
    Transcribes audio by wrapping PCM in a WAV container and sending to Groq's API.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        stt_config = models_meta.get("stt_providers", {}).get("groq", {}) or self.config
        
        self.api_key = stt_config.get("api_key") or os.getenv("GROQ_API_KEY")
        self.model = stt_config.get("model", "whisper-large-v3")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""
        if not self.api_key:
            logger.error("GroqSTT: No API key found. Please set GROQ_API_KEY in .env.")
            return ""

        try:
            # Wrap raw PCM 16-bit 16kHz mono audio in a WAV container in memory
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            
            wav_bytes = wav_buf.getvalue()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            files = {
                "file": ("audio.wav", wav_bytes, "audio/wav")
            }
            data = {
                "model": self.model,
                "language": "en"
            }
            
            response = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            return result.get("text", "").strip()
        except Exception as e:
            logger.exception(f"GroqSTT transcription failed: {e}")
            return ""
