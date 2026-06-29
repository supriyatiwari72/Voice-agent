import logging
import threading
import numpy as np
from typing import Dict, Any, Optional

from audio.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)

class AudioRecorder:
    """
    Production microphone recorder using sounddevice.

    Captures live audio from the default (or configured) input device
    at the sample rate and frame size defined in config. Each 30ms
    frame is normalized, pre-emphasized, converted to int16 PCM,
    and pushed into the shared AudioBuffer consumed by the NoiseWorker downstream.
    """

    def __init__(self, config: Dict[str, Any], input_buffer: AudioBuffer):
        self.config        = config
        self.input_buffer  = input_buffer
        self._active       = False
        self._stream       = None
        self._lock         = threading.Lock()

        audio_cfg = config.get("audio", {})
        self._sample_rate  = audio_cfg.get("sample_rate", 16000)
        self._channels     = audio_cfg.get("channels", 1)
        self._frame_ms     = audio_cfg.get("frame_duration_ms", 30)
        self._device       = audio_cfg.get("input_device", None)

        self._blocksize    = int(self._sample_rate * self._frame_ms / 1000)

        # WER reduction settings
        self._target_rms   = 0.15    # Target RMS for normalized audio
        self._pre_emphasis = 0.97    # Pre-emphasis filter coefficient

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Opens the microphone stream and begins capturing audio frames."""
        with self._lock:
            if self._active:
                return

            try:
                import sounddevice as sd
            except ImportError:
                raise RuntimeError(
                    "sounddevice is not installed. "
                    "Run: pip install sounddevice"
                )

            logger.info("Recorder Started")
            print("Recorder Started")

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._blocksize,
                device=self._device,
                callback=self._audio_callback,
                finished_callback=self._on_stream_finished,
            )
            self._stream.start()
            self._active = True

            logger.info(
                f"Microphone Opened — device={self._device or 'default'}, "
                f"rate={self._sample_rate}Hz, "
                f"blocksize={self._blocksize} samples ({self._frame_ms}ms)"
            )
            print("Microphone Opened")

    def stop_recording(self) -> None:
        """Stops the microphone stream gracefully."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"AudioRecorder: error closing stream: {e}")
                finally:
                    self._stream = None

        logger.info("Recorder Stopped")
        print("Recorder Stopped")

    def is_active(self) -> bool:
        """Returns True while the microphone stream is open and capturing."""
        return self._active

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """
        Called by sounddevice on each audio block (every 30ms).

        Applies RMS normalization + pre-emphasis filter, converts
        float32 samples in [-1.0, 1.0] to int16 PCM bytes, and
        pushes the result into the shared AudioBuffer.
        """
        if status:
            logger.debug(f"AudioRecorder stream status: {status}")

        if not self._active:
            return

        # Extract mono channel
        samples = indata[:, 0] if indata.ndim > 1 else indata

        # ── Pre-emphasis filter (high-pass) ───────────────────────────────
        # Amplifies high frequencies to improve consonant clarity for STT
        if len(samples) > 1:
            emphasized = np.empty_like(samples)
            emphasized[0] = samples[0]
            emphasized[1:] = samples[1:] - self._pre_emphasis * samples[:-1]
            samples = emphasized

        # ── RMS Normalization ─────────────────────────────────────────────
        # Boost quiet audio to a consistent level for better STT accuracy
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if 0.001 < rms < self._target_rms:
            gain = min(self._target_rms / rms, 4.0)
            samples = np.clip(samples * gain, -1.0, 1.0)

        # Convert float32 → int16 PCM
        pcm_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        pushed = self.input_buffer.push(pcm_bytes, timeout=0.0)
        if pushed:
            logger.debug(f"Audio Frame Captured — {len(pcm_bytes)} bytes")
        else:
            logger.debug("AudioRecorder: buffer full, frame dropped")

    def _on_stream_finished(self) -> None:
        """Called by sounddevice when the stream closes unexpectedly."""
        logger.info("AudioRecorder: stream finished.")

