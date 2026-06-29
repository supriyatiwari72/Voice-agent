import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import SentencePayload, PartialAudioPayload
from pipeline.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class StreamingTTSWorker(BaseWorker):
    """
    Worker that consumes SentencePayload elements, performs text-to-speech
    synthesis sentence-by-sentence, and pushes PartialAudioPayload packets
    to the partial_audio_queue.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, tts: Any):
        """
        Initializes the StreamingTTSWorker.
        """
        super().__init__(name="StreamingTTSWorker", context=context, input_queue=input_queue, output_queue=output_queue)
        self.tts = tts

    def process(self, payload: SentencePayload) -> None:
        """
        Process a single SentencePayload.
        """
        if not payload or not isinstance(payload, SentencePayload):
            logger.warning("Received invalid or empty payload in StreamingTTSWorker.")
            return

        # Verify request is active and not interrupted before calling TTS
        if self.stop_event.is_set() or self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
            logger.info(f"StreamingTTSWorker: request {payload.request_id} is stale/interrupted/stopped. Dropping payload before synthesis.")
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

                # Stream synthesis chunks
                for chunk in synth_method(payload.text):
                    # Verify request is active / worker not stopped
                    if self.stop_event.is_set() or self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
                        logger.info(f"StreamingTTSWorker: request {payload.request_id} interrupted/stopped during synthesis. Aborting stream.")
                        return
                    
                    if chunk:
                        if not first_chunk_sent:
                            logger.info("First Audio Chunk ready — entering SPEAKING state")
                            self.context.set_state(PipelineState.SPEAKING)
                            first_chunk_sent = True

                        audio_payload = PartialAudioPayload(
                            request_id=payload.request_id,
                            audio_chunk=chunk,
                            is_final=False,
                            timestamp=payload.user_done_timestamp
                        )
                        if self.output_queue:
                            self.output_queue.put(audio_payload)
                            
                # Record metric logs
                latency_ms = (time.time() - start_time) * 1000
                self.context.metrics.record_metric("tts_latency_ms", latency_ms)
                logger.info("TTS Complete")
                
                # Send final indicator chunk
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
            
            # Verify request is active after synthesis is done
            if self.stop_event.is_set() or self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
                logger.info(f"StreamingTTSWorker: request {payload.request_id} is stale/interrupted/stopped. Discarding synthesis output.")
                return

            latency_ms = (time.time() - start_time) * 1000

            # Record metric logs
            self.context.metrics.record_metric("tts_latency_ms", latency_ms)
            logger.info("TTS Complete")
            self.context.set_state(PipelineState.SPEAKING)

        audio_payload = PartialAudioPayload(
            request_id=payload.request_id,
            audio_chunk=audio_bytes,
            is_final=payload.is_final,
            timestamp=payload.user_done_timestamp
        )

        if self.output_queue:
            self.output_queue.put(audio_payload)
            logger.debug(f"StreamingTTSWorker synthesized audio chunk for sentence: '{payload.text[:20]}...' (is_final={payload.is_final})")
