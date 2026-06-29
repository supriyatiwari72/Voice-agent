import logging
import numpy as np
import os
import threading
import time
import wave
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationRecorder:
    """
    Records the entire conversation audio into a single WAV file until
    the conversation ends (triggered by "Goodbye!").

    Records both user speech (16kHz) and assistant TTS (24kHz) by
    resampling all audio to the recorder's native sample rate.
    """

    def __init__(self, session_id: str, output_dir: str = "audio/recordings"):
        self.session_id = session_id
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self._wav_file: Optional[wave.Wave_write] = None
        self._filepath: Optional[str] = None
        self._is_open = False
        self._sample_rate = 16000
        self._channels = 1
        self._sample_width = 2

    def open(self, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        with self._lock:
            if self._is_open:
                return
            os.makedirs(self.output_dir, exist_ok=True)
            self._filepath = os.path.join(
                self.output_dir,
                f"{self.session_id}.wav"
            )
            self._wav_file = wave.open(self._filepath, "wb")
            self._wav_file.setnchannels(channels)
            self._wav_file.setsampwidth(sample_width)
            self._wav_file.setframerate(sample_rate)
            self._sample_rate = sample_rate
            self._channels = channels
            self._sample_width = sample_width
            self._is_open = True
            logger.info(f"Conversation recording started: {self._filepath}")

    def write_audio(self, audio_bytes: bytes):
        if not audio_bytes:
            return
        with self._lock:
            if not self._is_open or self._wav_file is None:
                return
            try:
                self._wav_file.writeframes(audio_bytes)
            except Exception as e:
                logger.error(f"Failed to write audio to recording: {e}")

    def write_audio_from_rate(self, audio_bytes: bytes, source_rate: int):
        """
        Write audio bytes sampled at source_rate, resampling to the
        recorder's native sample_rate so both user (16kHz) and TTS
        (24kHz) can be recorded into the same file.
        """
        if not audio_bytes or source_rate == self._sample_rate:
            self.write_audio(audio_bytes)
            return
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            ratio = self._sample_rate / source_rate
            new_len = int(len(samples) * ratio)
            resampled = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.int16)
            self.write_audio(resampled.tobytes())
        except Exception as e:
            logger.error(f"Failed to resample audio from {source_rate}Hz to {self._sample_rate}Hz: {e}")

    def close(self):
        with self._lock:
            if not self._is_open:
                return
            self._is_open = False
            if self._wav_file is not None:
                try:
                    self._wav_file.close()
                except Exception as e:
                    logger.error(f"Failed to close recording file: {e}")
                finally:
                    self._wav_file = None
            logger.info(f"Conversation recording saved: {self._filepath}")

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath
