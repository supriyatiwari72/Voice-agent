import logging
import numpy as np
from typing import Any, Dict, Optional
from noise.base import BaseNoiseCanceller

logger = logging.getLogger(__name__)


class RNNoiseCanceller(BaseNoiseCanceller):
    """
    Noise cancellation using scipy high-pass filter + noisereduce.
    Removes low-frequency hum (fans, AC) while preserving speech.

    Falls back to bypass if scipy is unavailable.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self._biquad: Optional[Any] = None
        self._has_scipy = False
        self._has_noisereduce = False
        self._initialized = False

        try:
            from scipy import signal
            # 2nd-order high-pass Butterworth at 80Hz (removes hum, preserves speech)
            sos = signal.butter(4, 80 / (16000 / 2), btype="high", output="sos")
            self._sos = sos
            self._has_scipy = True
        except ImportError:
            self._has_scipy = False

        try:
            import noisereduce as nr
            self._nr = nr
            self._has_noisereduce = True
        except ImportError:
            self._has_noisereduce = False

        if self._has_scipy:
            logger.info("RNNoiseCanceller: scipy high-pass filter active (80Hz cutoff)")
            print("[Noise] scipy high-pass filter active")
        else:
            logger.warning("RNNoiseCanceller: scipy not installed — audio passing through unprocessed")
            print("[WARNING] scipy not installed — noise cancellation unavailable")

    def process(self, audio_data: bytes) -> bytes:
        if not audio_data or not self._has_scipy:
            return audio_data

        try:
            from scipy import signal
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            if len(audio_np) == 0:
                return audio_data

            # High-pass filter to remove low-frequency noise
            filtered = signal.sosfilt(self._sos, audio_np)

            # Optional: spectral noise reduction if noisereduce is available
            if self._has_noisereduce and len(audio_np) >= 2048:
                try:
                    filtered = self._nr.reduce_noise(
                        y=filtered, sr=16000, stationary=True,
                        prop_decrease=0.8, n_fft=512,
                    )
                except Exception:
                    pass

            cleaned = np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()
            return cleaned

        except Exception as e:
            logger.error(f"RNNoiseCanceller: processing failed, bypassing: {e}")
            return audio_data
