import logging
import threading
import numpy as np
from typing import Dict, Any, Optional
from audio.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Production audio player using sounddevice.

    Pops PCM int16 bytes from the shared output AudioBuffer and feeds them
    to the system's default output device in real time.  Each chunk is
    converted from raw int16 bytes → float32 [-1, 1] before being written
    to the sounddevice OutputStream.

    Configuration (config['audio']):
        sample_rate       (int,  default 24000) — Kokoro outputs 24 kHz
        channels          (int,  default 1)
        output_device     (int | None, default None → system default)
        player_chunk_ms   (int,  default 20)   — ms per write call
    """

    def __init__(self, config: Dict[str, Any], output_buffer: AudioBuffer):
        self.config = config
        self.output_buffer = output_buffer
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._lock = threading.Lock()
        self._skip_playback = False

        audio_cfg = config.get("audio", {})
        # Kokoro synthesizes at 24 kHz; allow override for other providers
        self._sample_rate  = audio_cfg.get("output_sample_rate",
                             audio_cfg.get("sample_rate", 24000))
        self._channels     = audio_cfg.get("output_channels",
                             audio_cfg.get("channels", 1))
        self._device       = audio_cfg.get("output_device", None)
        chunk_ms           = audio_cfg.get("player_chunk_ms", 20)
        self._blocksize    = int(self._sample_rate * chunk_ms / 1000)

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def start_playback(self) -> None:
        with self._lock:
            if self._active:
                return

            try:
                import sounddevice as sd
            except ImportError:
                raise RuntimeError(
                    "sounddevice is not installed. Run: pip install sounddevice"
                )

            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._blocksize,
                device=self._device,
            )
            self._stream.start()
            self._active = True

        logger.info(
            f"AudioPlayer started — device={self._device or 'default'}, "
            f"rate={self._sample_rate}Hz, blocksize={self._blocksize}"
        )

        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop_playback(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"AudioPlayer: error closing stream: {e}")
                finally:
                    self._stream = None

        logger.info("AudioPlayer stopped.")

    def interrupt(self) -> None:
        """Clear the output buffer and abort the audio stream immediately (barge-in)."""
        self._skip_playback = True
        self.output_buffer.clear()
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.abort()
                except Exception as e:
                    logger.warning(f"AudioPlayer: abort error: {e}")
        logger.debug("AudioPlayer: output buffer cleared and stream aborted (interrupt).")

    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Internal playback loop
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        """
        Continuously pops PCM int16 bytes from the output buffer and
        writes them to the sounddevice OutputStream.
        """
        first_chunk = True
        while self._active:
            chunk = self.output_buffer.pop(timeout=0.05)
            if chunk is None:
                continue

            if first_chunk:
                logger.info("First Audio Chunk — playback beginning")
                first_chunk = False

            self._write_pcm(chunk)

    def _write_pcm(self, pcm_bytes: bytes) -> None:
        """Convert int16 PCM bytes to float32 and write to OutputStream."""
        if not pcm_bytes:
            return

        # Auto gain normalization — boost quiet audio to a comfortable level
        samples_int16 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean((samples_int16 / 32768.0) ** 2)))
        target_rms = 0.15
        if 0.001 < rms < target_rms:
            gain = min(target_rms / rms, 4.0)
            samples_int16 = np.clip(samples_int16 * gain, -32768, 32767)

        samples = samples_int16 / 32768.0

        # Resample and log speaker reference samples for AEC
        if hasattr(self, "context") and self.context:
            try:
                from scipy import signal
                if self._sample_rate == 24000:
                    samples_16k = signal.resample_poly(samples, 2, 3)
                else:
                    samples_16k = samples
                with self.context.speaker_lock:
                    self.context.speaker_reference.extend(samples_16k)
            except Exception as e:
                logger.debug(f"AudioPlayer: error logging speaker reference: {e}")

        # Reshape to (frames, channels) expected by sounddevice
        if self._channels > 1:
            samples = samples.reshape(-1, self._channels)
        else:
            samples = samples.reshape(-1, 1)

        with self._lock:
            if self._stream is None:
                return
            if self._skip_playback:
                self._skip_playback = False
                logger.debug("AudioPlayer: skipped stale chunk after interrupt.")
                return
            if not self._stream.active:
                self._stream.start()
            try:
                self._stream.write(samples)
            except Exception as e:
                logger.warning(f"AudioPlayer: write error: {e}")
