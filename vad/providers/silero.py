import numpy as np
import torch
from silero_vad import load_silero_vad
from vad.base import BaseVAD

class SileroVAD(BaseVAD):
    """
    Concrete wrapper for Silero VAD (Voice Activity Detection) provider.
    Normalizes PCM audio bytes to tensors and queries the loaded model with configurable thresholds.
    """

    def __init__(self, config: dict):
        """
        Initializes the Silero VAD model once. Loads threshold settings.
        """
        self.config = config or {}
        
        # Resolve VAD configs from nested models_meta, root vad_providers, or fallback config
        models_meta = self.config.get("models_meta", {})
        vad_config = models_meta.get("vad_providers", {}).get("silero", {}) or \
                     self.config.get("vad_providers", {}).get("silero", {}) or \
                     self.config
        
        self.threshold = vad_config.get("threshold", 0.5)
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)

        # Load Silero VAD model JIT module
        self.model = load_silero_vad()
        print("Silero VAD Initialized")
        self._speech_detected_logged = False

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Calculates speech probability of the 16-bit 16kHz PCM audio chunk.
        Pads small chunks to meet the model's 512-sample minimum threshold.

        Args:
            audio_chunk (bytes): Raw PCM bytes.

        Returns:
            bool: True if the speech probability exceeds the configured threshold.
        """
        if not audio_chunk:
            return False

        # Convert 16-bit PCM bytes to normalized float32 numpy array [-1.0, 1.0]
        try:
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            return False

        if len(audio_np) == 0:
            return False

        # Pad to 512 samples if the input chunk is too short for the model
        if len(audio_np) < 512:
            audio_np = np.pad(audio_np, (0, 512 - len(audio_np)), 'constant')

        # Convert to PyTorch tensor with shape [1, N]
        tensor = torch.from_numpy(audio_np).unsqueeze(0)

        # Query model JIT module without gradients
        with torch.no_grad():
            speech_prob = self.model(tensor, self.sample_rate).item()

        is_speech_flag = speech_prob > self.threshold
        if is_speech_flag and not self._speech_detected_logged:
            print("Speech Detected")
            self._speech_detected_logged = True

        return is_speech_flag

    def get_speech_probability(self, audio_chunk: bytes) -> float:
        """
        Calculates the raw speech probability score from Silero VAD.
        """
        if not audio_chunk:
            return 0.0
        try:
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            return 0.0

        if len(audio_np) == 0:
            return 0.0

        if len(audio_np) < 512:
            audio_np = np.pad(audio_np, (0, 512 - len(audio_np)), 'constant')

        tensor = torch.from_numpy(audio_np).unsqueeze(0)

        with torch.no_grad():
            return float(self.model(tensor, self.sample_rate).item())
