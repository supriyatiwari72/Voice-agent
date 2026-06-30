import numpy as np
from noise.aec_base import BaseEchoCanceller

class CorrelationEchoCanceller(BaseEchoCanceller):
    """
    A temporary fallback leakage suppressor using cross-correlation and projection.
    This module is meant only as a fallback and can be easily swapped for an adaptive AEC.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}

    def process(self, mic_chunk: bytes, speaker_reference: list) -> bytes:
        if not mic_chunk or not speaker_reference:
            return mic_chunk

        # Convert raw mic bytes to float32 normalized samples
        try:
            mic_np = np.frombuffer(mic_chunk, dtype=np.int16).astype(np.float32)
        except ValueError:
            return mic_chunk

        if len(mic_np) == 0:
            return mic_chunk

        # Speaker reference holds 16kHz float32 samples
        speaker_samples = np.array(speaker_reference, dtype=np.float32)
        
        n_mic = len(mic_np)
        n_spk = len(speaker_samples)

        # We need enough speaker history to align (e.g. at least twice the mic chunk size)
        if n_spk < n_mic * 2 or np.max(np.abs(speaker_samples)) < 1e-3:
            return mic_chunk

        # Compute cross-correlation to find the delay of the echo path
        corr = np.correlate(speaker_samples, mic_np, mode='valid')
        best_delay_idx = np.argmax(np.abs(corr))

        # Extract the aligned speaker segment
        matched_speaker = speaker_samples[best_delay_idx : best_delay_idx + n_mic]

        if len(matched_speaker) == n_mic:
            spk_energy = np.dot(matched_speaker, matched_speaker)
            if spk_energy > 1e-4:
                # Least-squares scaling factor (alpha)
                alpha = np.dot(mic_np, matched_speaker) / spk_energy
                alpha = max(0.0, min(1.2, alpha))  # Prevent scaling artifacts
                
                # Subtract echo estimate
                cancelled_np = mic_np - alpha * matched_speaker
                
                # Attenuate frame heavily if cross-correlation is extremely high
                mic_energy = np.dot(mic_np, mic_np)
                if mic_energy > 1e-4:
                    r = np.abs(np.dot(mic_np, matched_speaker)) / np.sqrt(spk_energy * mic_energy)
                    if r > 0.7:
                        cancelled_np *= 0.05
                    elif r > 0.4:
                        cancelled_np *= 0.3

                # Clip and convert back to int16 bytes
                return np.clip(cancelled_np, -32768, 32767).astype(np.int16).tobytes()

        return mic_chunk
