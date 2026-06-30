import logging
import time
from typing import Any, Dict
from core.worker_base import BaseWorker
from core.payloads import SpeechPayload, PartialTranscriptPayload, StreamingAudioPayload
from pipeline.pipeline_state import PipelineState
from core.events import EventType, PartialTranscriptEvent, FinalTranscriptEvent

logger = logging.getLogger(__name__)


class STTWorker(BaseWorker):
    """
    Production STT Worker.
    Receives either:
    1. SpeechPayload (batch path): runs offline batch transcription and emits final PartialTranscriptPayload.
    2. StreamingAudioPayload (streaming path): streams raw chunks dynamically to streaming providers,
       or buffers them locally for batch engines, executing batch transcription on final chunk.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, stt: Any):
        super().__init__(
            name="STTWorker",
            context=context,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        self.stt = stt
        self._audio_buffers: Dict[str, bytes] = {}
        self._streaming_active = False
        self._active_stream_id = None
        self._last_emitted_transcripts: Dict[str, str] = {}

    def process(self, payload: Any) -> None:
        if not payload:
            logger.warning("Received invalid or empty payload in STTWorker.")
            return

        # ── Batch SpeechPayload Path (legacy and backward compatibility) ──
        if isinstance(payload, SpeechPayload):
            self._handle_speech_payload(payload)
            return

        # ── StreamingAudioPayload Path ─────────────────────────────────────
        if isinstance(payload, StreamingAudioPayload):
            self._handle_streaming_audio_payload(payload)
            return

        logger.warning(f"Unexpected payload type in STTWorker: {type(payload)}")

    def _handle_speech_payload(self, payload: SpeechPayload) -> None:
        if (self.stop_event.is_set() or 
            self.context.is_request_cancelled(payload.request_id) or
            self.context.interruption_event.is_set() or
            payload.request_id != self.context.get_active_request_id()):
            logger.info(f"STTWorker: request {payload.request_id} is stale/interrupted. Skipping.")
            return

        self.context.set_state(PipelineState.PROCESSING)
        start_time = time.time()

        audio_cfg = self.context.config.get("audio", {})
        sample_rate = audio_cfg.get("sample_rate", 16000)
        channels = audio_cfg.get("channels", 1)
        bytes_len = len(payload.audio)
        duration = bytes_len / (2 * channels * sample_rate) if (channels * sample_rate) > 0 else 0.0

        logger.info(f"[STT Audio Log] Request ID: {payload.request_id} | Duration: {duration:.2f}s | Size: {bytes_len} bytes")

        # Record user audio into conversation recorder
        recorder = getattr(self.context, "conversation_recorder", None)
        if recorder is not None:
            recorder.write_audio(payload.audio)

        # Batch transcription
        self._run_batch(payload, start_time, duration)

    def _handle_streaming_audio_payload(self, payload: StreamingAudioPayload) -> None:
        # Verify request is active
        if (self.stop_event.is_set() or 
            self.context.is_request_cancelled(payload.request_id) or
            self.context.interruption_event.is_set() or
            payload.request_id != self.context.get_active_request_id()):
            self._audio_buffers.pop(payload.request_id, None)
            self._last_emitted_transcripts.pop(payload.request_id, None)
            if self._active_stream_id == payload.request_id:
                self._stop_provider_stream()
            return

        # Record user audio frame into conversation recorder
        recorder = getattr(self.context, "conversation_recorder", None)
        if recorder is not None and payload.audio_chunk:
            recorder.write_audio(payload.audio_chunk)

        # Determine capabilities of the STT provider
        supports_stream = False
        if hasattr(self.stt, "supports_streaming_audio"):
            supports_stream = self.stt.supports_streaming_audio()
        else:
            supports_stream = (
                hasattr(self.stt, "start_stream")
                and hasattr(self.stt, "stream_audio")
                and hasattr(self.stt, "stop_stream")
            )

        if supports_stream:
            # ── Stream audio in real-time to provider ───────────────────
            if not self._streaming_active or self._active_stream_id != payload.request_id:
                self._start_provider_stream(payload)

            if payload.audio_chunk:
                self.stt.stream_audio(payload.audio_chunk)

            if payload.is_final:
                self._stop_provider_stream()
        else:
            # ── Buffer locally for batch execution at EOS ───────────────
            self.context.set_state(PipelineState.LISTENING)
            self._audio_buffers[payload.request_id] = (
                self._audio_buffers.get(payload.request_id, b"")
                + payload.audio_chunk
            )
            
            if payload.is_final:
                complete_audio = self._audio_buffers.pop(payload.request_id, b"")
                self._run_batch_from_stream(payload.request_id, complete_audio, payload.timestamp)

    def _start_provider_stream(self, payload: StreamingAudioPayload) -> None:
        self.context.set_state(PipelineState.PROCESSING)
        self._streaming_active = True
        self._active_stream_id = payload.request_id
        self._last_emitted_transcripts[payload.request_id] = ""

        def on_transcript(chunk: str, is_final: bool) -> None:
            if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
                return

            # Apply transcript stabilization
            cleaned_new = chunk.strip()
            cleaned_old = self._last_emitted_transcripts.get(payload.request_id, "").strip()
            
            if not cleaned_new:
                if is_final:
                    # Emit final empty chunk to propagate EOS
                    out = PartialTranscriptPayload(
                        request_id=payload.request_id,
                        text_chunk="",
                        is_final=True,
                        timestamp=payload.timestamp
                    )
                    if self.output_queue:
                        self.output_queue.put(out)
                return

            words_new = cleaned_new.split()
            words_old = cleaned_old.split()

            if is_final or len(words_new) > len(words_old) or (len(words_new) == len(words_old) and words_new != words_old):
                self._last_emitted_transcripts[payload.request_id] = cleaned_new
                
                # Performance metric check for first transcript latency
                if self.context.streaming_context.check_and_record_first_transcript(payload.request_id):
                    first_stt_lat = (time.time() - payload.timestamp) * 1000
                    self.context.metrics.record_metric("first_partial_transcript_ms", first_stt_lat)

                # Emit Typed Events
                if is_final:
                    self.context.trigger_event(
                        EventType.ERROR_EVENT, 
                        FinalTranscriptEvent(request_id=payload.request_id, text=cleaned_new, timestamp=time.time())
                    )
                else:
                    self.context.trigger_event(
                        EventType.ERROR_EVENT,
                        PartialTranscriptEvent(request_id=payload.request_id, text_chunk=cleaned_new[len(cleaned_old):].strip(), timestamp=time.time())
                    )

                out = PartialTranscriptPayload(
                    request_id=payload.request_id,
                    text_chunk=cleaned_new[len(cleaned_old):].strip() + " ", 
                    is_final=is_final,
                    timestamp=payload.timestamp
                )
                if is_final:
                    print(f"\n[You]\n{cleaned_new}\n", flush=True)
                    out.text_chunk = cleaned_new
                
                if self.output_queue:
                    self.output_queue.put(out)

        self.stt.start_stream(payload.request_id, on_transcript)

    def _stop_provider_stream(self) -> None:
        if self._streaming_active:
            try:
                self.stt.stop_stream()
            except Exception as e:
                logger.error(f"Error stopping provider stream: {e}")
            self._streaming_active = False
            self._active_stream_id = None

    def _is_hallucination(self, text: str, duration: float) -> bool:
        cleaned = text.strip().lower().rstrip(".,!?")
        if not cleaned:
            return True
        
        hallucinations = {
            # Common Whisper static/noise hallucinations
            "thank you", "thank you very much", "thank you so much",
            "you're welcome", "you", "thanks", "thanks for watching",
            "subtitles by", "bye", "goodbye", "see you",
            # Common TTS echo hallucinations (picked up from speaker)
            "it's good", "it's good it's good", "beyond that",
            "have to put one on it", "please", "joke",
            "oh my goodness", "what do you think about this",
            # Whisper silence hallucinations
            ".", "..", "...", "okay", "ok", "um", "uh", "hmm",
        }
        
        if cleaned in hallucinations and duration < 2.5:
            return True
        return False



    def _run_batch_from_stream(self, request_id: str, audio: bytes, timestamp: float) -> None:
        self.context.set_state(PipelineState.PROCESSING)
        start_time = time.time()
        
        audio_cfg = self.context.config.get("audio", {})
        sample_rate = audio_cfg.get("sample_rate", 16000)
        channels = audio_cfg.get("channels", 1)
        bytes_len = len(audio)
        duration = bytes_len / (2 * channels * sample_rate) if (channels * sample_rate) > 0 else 0.0
        
        text = self.stt.transcribe(audio)
        
        latency_ms = (time.time() - start_time) * 1000
        self.context.metrics.record_metric("stt_latency_ms", latency_ms)
        
        if self._is_hallucination(text, duration):
            logger.warning(f"STTWorker: Discarding hallucinated transcript '{text}' (duration={duration:.2f}s)")
            self.context.set_state(PipelineState.IDLE)
            return

        print(f"\n[You]\n{text}\n", flush=True)
        
        if (self.stop_event.is_set() or 
            self.context.is_request_cancelled(request_id) or
            self.context.interruption_event.is_set() or
            request_id != self.context.get_active_request_id()):
            return
            
        if self.context.streaming_context.check_and_record_first_transcript(request_id):
            first_stt_lat = (time.time() - timestamp) * 1000
            self.context.metrics.record_metric("first_partial_transcript_ms", first_stt_lat)

        self.context.trigger_event(
            EventType.ERROR_EVENT, 
            FinalTranscriptEvent(request_id=request_id, text=text, timestamp=time.time())
        )
            
        out = PartialTranscriptPayload(
            request_id=request_id,
            text_chunk=text,
            is_final=True,
            timestamp=timestamp
        )
        if self.output_queue:
            self.output_queue.put(out)

    def _run_batch(self, payload: SpeechPayload, start_time: float, duration: float) -> None:
        text = self.stt.transcribe(payload.audio)
        latency_seconds = time.time() - start_time
        latency_ms = latency_seconds * 1000
        self.context.metrics.record_metric("stt_latency_ms", latency_ms)

        rtf = latency_seconds / duration if duration > 0.0 else 0.0
        logger.info(f"STT Latency: {latency_ms:.0f} ms | RTF: {rtf:.2f}")

        if self._is_hallucination(text, duration):
            logger.warning(f"STTWorker: Discarding hallucinated transcript '{text}' (duration={duration:.2f}s)")
            self.context.set_state(PipelineState.IDLE)
            return

        print(f"\n[You]\n{text}\n", flush=True)

        if (self.stop_event.is_set() or 
            self.context.is_request_cancelled(payload.request_id) or
            self.context.interruption_event.is_set() or
            payload.request_id != self.context.get_active_request_id()):
            return

        if self.context.streaming_context.check_and_record_first_transcript(payload.request_id):
            first_stt_lat = (time.time() - payload.user_done_timestamp) * 1000
            self.context.metrics.record_metric("first_partial_transcript_ms", first_stt_lat)

        self.context.trigger_event(
            EventType.ERROR_EVENT, 
            FinalTranscriptEvent(request_id=payload.request_id, text=text, timestamp=time.time())
        )

        out = PartialTranscriptPayload(
            request_id=payload.request_id,
            text_chunk=text,
            is_final=True,
            timestamp=payload.user_done_timestamp,
        )
        if self.output_queue:
            self.output_queue.put(out)

    def handle_error(self, e: Exception) -> None:
        logger.error(f"STTWorker error: {e}")
        self._stop_provider_stream()
        self._audio_buffers.clear()
        self._last_emitted_transcripts.clear()
        self.context.set_state(PipelineState.IDLE)
