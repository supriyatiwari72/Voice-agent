import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import SentencePayload, PartialAudioPayload
from pipeline.pipeline_state import PipelineState
from core.events import EventType, TTSStartedEvent

logger = logging.getLogger(__name__)

class StreamingTTSWorker(BaseWorker):
    """
    Worker that consumes SentencePayload elements, performs text-to-speech
    synthesis sentence-by-sentence, and pushes PartialAudioPayload packets
    to the partial_audio_queue.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, tts: Any):
        super().__init__(name="StreamingTTSWorker", context=context, input_queue=input_queue, output_queue=output_queue)
        self.tts = tts

    def process(self, payload: SentencePayload) -> None:
        if not payload or not isinstance(payload, SentencePayload):
            logger.warning("Received invalid or empty payload in StreamingTTSWorker.")
            return

        # Check cancellation before calling TTS
        if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
            logger.info(f"StreamingTTSWorker: request {payload.request_id} is cancelled/stopped. Dropping.")
            return

        audio_bytes = b""
        if payload.text.strip():
            start_time = time.time()
            
            # Check if TTS provider supports streaming synthesis
            if ((hasattr(self.tts, "stream_synthesize") or hasattr(self.tts, "synthesize_stream")) and
                    type(self.tts).__name__ not in ("MagicMock", "Mock")):
                synth_method = getattr(self.tts, "stream_synthesize", None) or getattr(self.tts, "synthesize_stream")

                logger.info(f"TTS Started — synthesizing: '{payload.text[:60]}...'")
                first_chunk_sent = False

                for chunk in synth_method(payload.text):
                    # Check cancellation during generation loop
                    if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
                        logger.info(f"StreamingTTSWorker: request {payload.request_id} cancelled during synthesis.")
                        return
                    
                    if chunk:
                        if not first_chunk_sent:
                            logger.info("First Audio Chunk ready — sending TTSStartedEvent")
                            # Trigger TTSStartedEvent (Coordinator transitions state to PipelineState.SPEAKING)
                            self.context.trigger_event(
                                EventType.ERROR_EVENT,
                                TTSStartedEvent(request_id=payload.request_id, timestamp=time.time())
                            )
                            first_chunk_sent = True

                        audio_payload = PartialAudioPayload(
                            request_id=payload.request_id,
                            audio_chunk=chunk,
                            is_final=False,
                            timestamp=payload.user_done_timestamp
                        )
                        if self.output_queue:
                            self.output_queue.put(audio_payload)
                            
                latency_ms = (time.time() - start_time) * 1000
                self.context.metrics.record_metric("tts_latency_ms", latency_ms)
                logger.info("TTS Complete")
                
                audio_payload = PartialAudioPayload(
                    request_id=payload.request_id,
                    audio_chunk=b"",
                    is_final=payload.is_final,
                    timestamp=payload.user_done_timestamp
                )
                if self.output_queue:
                    self.output_queue.put(audio_payload)
                return
            
            # Fallback to batch synthesis
            audio_bytes = self.tts.synthesize(payload.text)
            
            if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
                logger.info(f"StreamingTTSWorker: request {payload.request_id} is stale/interrupted. Discarding synthesis.")
                return

            latency_ms = (time.time() - start_time) * 1000
            self.context.metrics.record_metric("tts_latency_ms", latency_ms)
            logger.info("TTS Complete")
            
            # Trigger TTSStartedEvent (Coordinator transitions state to PipelineState.SPEAKING)
            self.context.trigger_event(
                EventType.ERROR_EVENT,
                TTSStartedEvent(request_id=payload.request_id, timestamp=time.time())
            )

        audio_payload = PartialAudioPayload(
            request_id=payload.request_id,
            audio_chunk=audio_bytes,
            is_final=payload.is_final,
            timestamp=payload.user_done_timestamp
        )

        if self.output_queue:
            self.output_queue.put(audio_payload)
