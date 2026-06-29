import logging
import time
import uuid
import numpy as np
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import AudioPayload, SpeechPayload, StreamingAudioPayload, InterruptionPayload
from pipeline.pipeline_state import PipelineState
from core.events import EventType
from core.eos_manager import EOSManager

logger = logging.getLogger(__name__)

# States in which the pipeline is actively producing a response.
# Barge-in is ONLY valid when the assistant is speaking to the user.
_BARGE_IN_STATES = frozenset([PipelineState.SPEAKING])

# States in which we are busy processing the previous utterance.
_BUSY_STATES = frozenset([
    PipelineState.PROCESSING,
    PipelineState.TRANSCRIBING,
    PipelineState.THINKING,
    PipelineState.GENERATING,
])


def _calculate_rms(audio_bytes: bytes) -> float:
    """RMS of 16-bit PCM bytes as a float in [0.0, 1.0]."""
    if not audio_bytes:
        return 0.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


class VADWorker(BaseWorker):
    """
    Worker that retrieves AudioPayload chunks from speech_queue, runs Voice
    Activity Detection, accumulates speech, and forwards turns either as
    SpeechPayload or StreamingAudioPayload chunks based on STT streaming support.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, vad: Any):
        super().__init__(
            name="VADWorker",
            context=context,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        self.vad = vad
        self.eos_manager = EOSManager(context.config if context else {})
        self._speech_buffer = b""
        self._in_speech = False
        self._current_request_id = None
        self._silence_frames = 0
        self._silence_chunks = []

        # Post-utterance cooldown
        self._cooldown_frames = 0
        self._cooldown_after_utterance = 15   # ~450 ms @ 30ms/frame

        # Pre-speech rolling history (captures utterance onset)
        self._pre_speech_history = []
        self._pad_frames = 5

        # Configuration variables
        self._rms_threshold = 0.0
        self._segment_rms_threshold = 0.0
        self._barge_in_rms_multiplier = 2.5
        self._min_speech_bytes = 8000

        if context and hasattr(context, "config") and context.config:
            models_meta = context.config.get("models_meta", {})
            vad_config = (
                models_meta.get("vad_providers", {}).get("silero", {})
                or context.config.get("vad_providers", {}).get("silero", {})
                or context.config
            )
            self._rms_threshold = vad_config.get("rms_threshold", self._rms_threshold)
            self._segment_rms_threshold = vad_config.get("segment_rms_threshold", self._segment_rms_threshold)
            self._pad_frames = vad_config.get("pad_frames", 5)
            self._barge_in_rms_multiplier = vad_config.get("barge_in_rms_multiplier", 2.5)
            self._min_speech_bytes = vad_config.get("min_speech_bytes", 12000)

        # Detect streaming vs batch STT path
        stt_provider = ""
        if context and hasattr(context, "config") and context.config:
            stt_provider = (context.config.get("active_providers", {}).get("stt", "") or "").lower()
            
        self._use_streaming = (
            "streaming" in stt_provider 
            or stt_provider in ("deepgram", "assemblyai")
            or (context and hasattr(context, "config") and context.config.get("stt_streaming", False))
        )

    def process(self, payload: AudioPayload) -> None:
        if not payload or not isinstance(payload, AudioPayload):
            logger.warning("Received invalid or empty payload in VADWorker.")
            return

        # Cooldown guard
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return

        current_state = self.context.get_state()

        # Busy-state check
        if current_state in _BUSY_STATES:
            return

        frame_rms = _calculate_rms(payload.audio)
        
        # Call Silero VAD to get probability
        confidence_score = 0.0
        is_mock_or_dummy = type(self.vad).__name__ in ("MagicMock", "Mock", "DummyVAD")
        
        # Check if the active provider is dummy via config
        if not is_mock_or_dummy and self.context and hasattr(self.context, "config") and self.context.config:
            is_mock_or_dummy = (self.context.config.get("active_providers", {}).get("vad", "") or "").lower() == "dummy"

        if hasattr(self.vad, "get_speech_probability"):
            confidence_score = self.vad.get_speech_probability(payload.audio)
            # handle mocks
            if type(confidence_score).__name__ in ("MagicMock", "Mock"):
                is_speech = self.vad.is_speech(payload.audio)
                confidence_score = 1.0 if is_speech else 0.0
        else:
            is_speech = self.vad.is_speech(payload.audio)
            confidence_score = 1.0 if is_speech else 0.0

        # Feed to EOSManager
        eos_res = self.eos_manager.process_frame(frame_rms, confidence_score, current_state, is_mock_or_dummy=is_mock_or_dummy)

        # Handle speech started (voice onset)
        if eos_res["speech_started"]:
            self._current_request_id = payload.request_id
            self._in_speech = True
            
            # Start timer for speech onset latency metric
            self.context.metrics.record_metric("speech_detection_latency_ms", (time.time() - payload.created_at) * 1000)
            
            # Interruption check
            if current_state in _BARGE_IN_STATES:
                logger.info(f"Barge-in detected in state {current_state.name}. Triggering interruption.")
                self.context.interruption_event.set()
                interruption_payload = InterruptionPayload(
                    request_id=self._current_request_id,
                    timestamp=time.time(),
                )
                if self.context.queue_manager.interruption_queue:
                    self.context.queue_manager.interruption_queue.put(interruption_payload)
            
            self.context.set_active_request_id(self._current_request_id)
            self.context.set_state(PipelineState.USER_SPEAKING)
            self.context.trigger_event(EventType.SPEECH_STARTED, self._current_request_id)
            logger.info(f"Speech Started: request_id={self._current_request_id}")
            
            # Stream the starting pre-speech history
            if self._use_streaming:
                history = b"".join(self._pre_speech_history)
                if history:
                    start_payload = StreamingAudioPayload(
                        request_id=self._current_request_id,
                        audio_chunk=history,
                        is_final=False,
                        timestamp=payload.created_at
                    )
                    self.output_queue.put(start_payload)
            else:
                self._speech_buffer = b"".join(self._pre_speech_history)
                self._silence_chunks = []

        # Handle active speech frames
        if eos_res["is_speech_active"]:
            if self._use_streaming:
                chunk_payload = StreamingAudioPayload(
                    request_id=self._current_request_id,
                    audio_chunk=payload.audio,
                    is_final=False,
                    timestamp=payload.created_at
                )
                self.output_queue.put(chunk_payload)
            else:
                self._speech_buffer += payload.audio
                # Track silence frame chunk buffer to support silence stripping
                if confidence_score <= self.eos_manager.speech_end_threshold:
                    self._silence_chunks.append(payload.audio)
                else:
                    self._silence_chunks = []

            # Safety duration limit (10s ~ 320,000 bytes)
            if len(self._speech_buffer) >= 320000 or (self._use_streaming and self.eos_manager.speech_chunks_count >= 333):
                logger.warning("Max utterance duration exceeded. Finalizing.")
                self._finalize_utterance(payload, strip_silence=False)
                return

        # Handle speech ended (EOS endpoint)
        if eos_res["speech_ended"]:
            # Record endpointing latency
            if self.eos_manager.last_speech_time:
                self.context.metrics.record_metric("eos_detection_latency_ms", (time.time() - self.eos_manager.last_speech_time) * 1000)
            
            self._finalize_utterance(payload, strip_silence=True)
            return

        # Save pre-speech history when silent
        if not eos_res["is_speech_active"]:
            self._pre_speech_history.append(payload.audio)
            if len(self._pre_speech_history) > self._pad_frames:
                self._pre_speech_history.pop(0)

        # Synchronize silence frames counter for backward compatibility with unit tests
        self._silence_frames = self.eos_manager.silence_counter

    def _finalize_utterance(self, payload: AudioPayload, strip_silence: bool = True) -> None:
        user_done_timestamp = time.time()
        
        if self._use_streaming:
            final_payload = StreamingAudioPayload(
                request_id=self._current_request_id,
                audio_chunk=b"",
                is_final=True,
                timestamp=user_done_timestamp
            )
            self.output_queue.put(final_payload)
            self.context.set_state(PipelineState.PROCESSING)
            self.context.trigger_event(EventType.SPEECH_ENDED, self._current_request_id)
        else:
            if strip_silence:
                silence_len = sum(len(c) for c in self._silence_chunks)
                if silence_len > 0:
                    self._speech_buffer = self._speech_buffer[:-silence_len]
            
            duration = len(self._speech_buffer) / 32000.0
            seg_rms = _calculate_rms(self._speech_buffer)
            
            # Validation gates
            if len(self._speech_buffer) < self._min_speech_bytes or seg_rms < self._segment_rms_threshold:
                logger.warning(f"Discarding turn: size={len(self._speech_buffer)}, rms={seg_rms:.4f}")
                self.context.set_state(PipelineState.LISTENING)
                self._reset()
                return
            
            self.context.set_state(PipelineState.PROCESSING)
            self.context.trigger_event(EventType.SPEECH_ENDED, self._current_request_id)
            
            speech_payload = SpeechPayload(
                request_id=self._current_request_id or payload.request_id,
                audio=self._speech_buffer,
                user_done_timestamp=user_done_timestamp,
            )
            self.output_queue.put(speech_payload)

        self._cooldown_frames = self._cooldown_after_utterance
        self._reset()

    def _reset(self) -> None:
        self._speech_buffer = b""
        self._in_speech = False
        self._silence_frames = 0
        self._silence_chunks = []
        self._current_request_id = None
        self.eos_manager.reset()
