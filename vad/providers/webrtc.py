import logging
import numpy as np
from vad.base import BaseVAD

logger = logging.getLogger(__name__)

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


class WebRTCVAD(BaseVAD):
    """
    Concrete Voice Activity Detection (VAD) provider using WebRTC.
    - If 'webrtcvad' is installed, wraps the native C extension.
    - If not, falls back to a custom pure-Python energy and zero-crossing frequency VAD.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        
        # Resolve configurations
        webrtc_config = (
            models_meta.get("vad_providers", {}).get("webrtc", {})
            or self.config.get("vad_providers", {}).get("webrtc", {})
            or {}
        )
        
        self.aggressiveness = int(webrtc_config.get("aggressiveness", 3))
        # Ensure aggressiveness is within webrtcvad's limits [0, 3]
        self.aggressiveness = max(0, min(3, self.aggressiveness))
        
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)
        
        # Setup native VAD if available
        if WEBRTCVAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(self.aggressiveness)
                logger.info(f"WebRTC VAD: Native engine initialized (aggressiveness={self.aggressiveness})")
            except Exception as e:
                logger.warning(f"WebRTC VAD: Failed to initialize native Vad. Falling back to pure Python: {e}")
                self.vad = None
        else:
            logger.info("WebRTC VAD: Native webrtcvad not installed. Using pure-Python ZCR + Energy fallback.")
            self.vad = None

        # Fallback thresholds scaled by aggressiveness
        # Higher aggressiveness means higher threshold (less sensitive to noise, requires clearer speech)
        if self.aggressiveness == 0:
            self._energy_threshold = 0.010
            self._zcr_min = 0.02
            self._zcr_max = 0.40
        elif self.aggressiveness == 1:
            self._energy_threshold = 0.018
            self._zcr_min = 0.025
            self._zcr_max = 0.38
        elif self.aggressiveness == 2:
            self._energy_threshold = 0.028
            self._zcr_min = 0.03
            self._zcr_max = 0.35
        else: # Mode 3 (default)
            self._energy_threshold = 0.040
            self._zcr_min = 0.035
            self._zcr_max = 0.32

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Processes the raw audio chunk to detect if voice is active.
        """
        if not audio_chunk:
            return False

        # If native WebRTC VAD is available, use it
        if self.vad is not None:
            try:
                # webrtcvad requires 16-bit PCM mono frames of 10, 20, or 30 ms.
                # A 30ms frame at 16000Hz is 480 samples = 960 bytes.
                # Handle arbitrary chunk lengths by dividing into 10ms slices if needed.
                if len(audio_chunk) % 2 != 0:
                    audio_chunk = audio_chunk[:len(audio_chunk) - (len(audio_chunk) % 2)]

                num_samples = len(audio_chunk) // 2
                duration_ms = (num_samples / self.sample_rate) * 1000
                rounded_duration = int(round(duration_ms))

                if rounded_duration in (10, 20, 30):
                    expected_bytes = int(rounded_duration * self.sample_rate / 1000) * 2
                    if len(audio_chunk) == expected_bytes:
                        return self.vad.is_speech(audio_chunk, self.sample_rate)

                # Slice chunks into 10ms frames if the input length is non-standard
                frame_len_10ms = int(10 * self.sample_rate / 1000) * 2
                i = 0
                while i + frame_len_10ms <= len(audio_chunk):
                    slice_bytes = audio_chunk[i:i + frame_len_10ms]
                    if self.vad.is_speech(slice_bytes, self.sample_rate):
                        return True
                    i += frame_len_10ms
                return False

            except Exception as e:
                logger.debug(f"Native webrtcvad error: {e}. Falling back to pure Python.")

        # Pure-Python fallback using Short-Time Energy & Zero Crossing Rate (ZCR)
        try:
            samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            return False

        if len(samples) == 0:
            return False

        # 1. Calculate RMS energy
        rms = float(np.sqrt(np.mean(samples ** 2)))

        # 2. Calculate Zero Crossing Rate (ZCR)
        # ZCR counts the frequency of sign changes. Voiced speech (300-3000Hz) has stable ZCR,
        # while ambient low hum has extremely low ZCR, and white noise has high ZCR (>0.4).
        sign_changes = np.sum(np.diff(np.sign(samples)) != 0)
        zcr = float(sign_changes / len(samples))

        # Classify as speech if energy is above the floor and crossing rate matches human vocal frequencies
        is_active = (rms >= self._energy_threshold) and (self._zcr_min <= zcr <= self._zcr_max)
        return is_active
