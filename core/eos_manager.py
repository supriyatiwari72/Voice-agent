import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EOSManager:
    """
    Dedicated End-of-Speech (EOS) Manager.
    Decides when speech starts and ends using multiple signals.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        
        # Load settings from config, defaulting to standard human conversational parameters
        eos_config = models_meta.get("eos", {}) or self.config.get("eos", {})
        
        # Hysteresis thresholds
        self.speech_start_threshold = float(eos_config.get("speech_start_threshold", 0.60))
        self.speech_end_threshold = float(eos_config.get("speech_end_threshold", 0.30))
        
        # Consecutive silent frames and timing thresholds
        vad_providers = models_meta.get("vad_providers", {}) or self.config.get("vad_providers", {})
        silero_config = vad_providers.get("silero", {}) if isinstance(vad_providers, dict) else {}
        if not isinstance(silero_config, dict):
            silero_config = {}

        self.consecutive_silence_frames = int(
            eos_config.get("consecutive_silence_frames")
            or silero_config.get("max_silence_frames")
            or 20
        )
        self.min_speech_duration_ms = float(eos_config.get("min_speech_duration_ms", 300.0))
        self.silence_timeout_ms = float(eos_config.get("silence_timeout_ms", 800.0))
        
        # Dynamic threshold scaling multiplier during Friday playback (barge-in echo suppression)
        self.barge_in_rms_multiplier = float(eos_config.get("barge_in_rms_multiplier", 5.0))
        
        audio_cfg = self.config.get("audio", {})
        self.frame_duration_ms = float(audio_cfg.get("frame_duration_ms", 30))
        
        # Adaptive noise floor tracking
        self.noise_floor = float(silero_config.get("rms_threshold") or 0.025)
        self._rms_history = []
        self._max_history = 100
        
        self.reset()

    def reset(self) -> None:
        """Resets the state of the manager for a new conversational turn."""
        self.in_speech = False
        self.silence_counter = 0
        self.speech_start_time = None
        self.last_speech_time = None
        self.speech_chunks_count = 0

    def process_frame(self, frame_rms: float, vad_confidence: float, current_state: Any, is_mock_or_dummy: bool = False) -> Dict[str, Any]:
        """
        Processes an audio frame's energy and VAD confidence, returning transitions.
        
        Returns:
            Dict containing:
              - is_speech_active (bool)
              - speech_started (bool)
              - speech_ended (bool)
              - noise_floor (float)
        """
        # 1. Update the noise floor dynamically using low-confidence silence frames
        if vad_confidence < 0.1:
            self._rms_history.append(frame_rms)
            if len(self._rms_history) > self._max_history:
                self._rms_history.pop(0)
            
            # Simple, stable noise floor: mean of the lowest 50% RMS values
            if len(self._rms_history) >= 10:
                sorted_rms = sorted(self._rms_history)
                half_len = len(sorted_rms) // 2
                self.noise_floor = max(0.005, min(0.050, sum(sorted_rms[:half_len]) / half_len))

        # 2. Adjust thresholds based on the pipeline's active playback state
        # In a SPEAKING state, Friday is playing audio, so we scale up energy gating and speech start thresholds.
        from pipeline.pipeline_state import PipelineState
        is_friday_speaking = (current_state == PipelineState.SPEAKING)
        
        if is_mock_or_dummy:
            energy_gate = 0.0
            start_threshold = self.speech_start_threshold
        else:
            if is_friday_speaking:
                energy_gate = self.noise_floor * self.barge_in_rms_multiplier
                start_threshold = min(0.95, self.speech_start_threshold * 1.3)
            else:
                energy_gate = self.noise_floor * 1.0
                start_threshold = self.speech_start_threshold

        # 3. Check VAD and energy gates
        is_frame_speech = False
        if self.in_speech:
            is_frame_speech = (vad_confidence > self.speech_end_threshold) and (frame_rms >= energy_gate * 0.7)
        else:
            is_frame_speech = (vad_confidence > start_threshold) and (frame_rms >= energy_gate)

        speech_started = False
        speech_ended = False

        if is_frame_speech:
            self.silence_counter = 0
            self.last_speech_time = time.time()
            self.speech_chunks_count += 1

            if not self.in_speech:
                self.in_speech = True
                self.speech_start_time = time.time()
                speech_started = True
                logger.info("EOSManager: Voice onset start detected.")
        else:
            if self.in_speech:
                self.silence_counter += 1
                
                # Check silence end timeout / consecutive frames met
                elapsed_silence_ms = self.silence_counter * self.frame_duration_ms
                consecutive_silence_met = self.silence_counter >= self.consecutive_silence_frames
                timeout_met = elapsed_silence_ms >= self.silence_timeout_ms
                
                if consecutive_silence_met or timeout_met:
                    speech_duration_ms = (time.time() - self.speech_start_time) * 1000
                    
                    if is_mock_or_dummy or speech_duration_ms >= self.min_speech_duration_ms:
                        self.in_speech = False
                        speech_ended = True
                        logger.info(
                            f"EOSManager: Endpoint detected. "
                            f"Speech duration: {speech_duration_ms:.0f}ms, "
                            f"Silence: {elapsed_silence_ms:.0f}ms."
                        )
                    else:
                        # Reject accidental clicks / static bursts
                        logger.info(
                            f"EOSManager: Speech burst rejected (duration: {speech_duration_ms:.0f}ms "
                            f"< {self.min_speech_duration_ms:.0f}ms)."
                        )
                        self.reset()

        return {
            "is_speech_active": self.in_speech,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "noise_floor": self.noise_floor
        }
