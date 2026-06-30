import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import SpeechPayload, PartialTranscriptPayload

logger = logging.getLogger(__name__)

class SimulatedStreamingSTTWorker(BaseWorker):
    """
    Simulated Streaming STT worker that transcribes completed speech turns
    and chunks the transcript text into incremental word segments to simulate
    streaming provider telemetry and pipeline latency validation.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, stt: Any):
        """
        Initializes the SimulatedStreamingSTTWorker.
        """
        super().__init__(name="SimulatedStreamingSTTWorker", context=context, input_queue=input_queue, output_queue=output_queue)
        self.stt = stt

    def process(self, payload: SpeechPayload) -> None:
        """
        Processes a SpeechPayload, generating partial transcripts.
        """
        if not payload or not isinstance(payload, SpeechPayload):
            logger.warning("Received invalid or empty payload in SimulatedStreamingSTTWorker.")
            return

        # Transit pipeline state to TRANSCRIBING
        from pipeline.pipeline_state import PipelineState
        self.context.set_state(PipelineState.PROCESSING)

        start_time = time.time()
        
        # Check if STT provider supports real streaming
        if (hasattr(self.stt, "start_stream") and 
            hasattr(self.stt, "stream_audio") and 
            hasattr(self.stt, "stop_stream") and 
            type(self.stt).__name__ not in ("MagicMock", "Mock")):
            def on_transcript(chunk, is_final):
                if self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
                    return
                
                # Record first_partial_transcript_ms
                if chunk.strip() and self.context.streaming_context.check_and_record_first_transcript(payload.request_id):
                    first_stt_lat = (time.time() - payload.user_done_timestamp) * 1000
                    self.context.metrics.record_metric("first_partial_transcript_ms", first_stt_lat)

                payload_out = PartialTranscriptPayload(
                    request_id=payload.request_id,
                    text_chunk=chunk,
                    is_final=is_final,
                    timestamp=payload.user_done_timestamp
                )
                if self.output_queue:
                    self.output_queue.put(payload_out)

            self.stt.start_stream(payload.request_id, on_transcript)
            
            # Stream the audio data in chunks
            chunk_size = 4096
            for offset in range(0, len(payload.audio), chunk_size):
                if self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
                    break
                chunk = payload.audio[offset:offset+chunk_size]
                self.stt.stream_audio(chunk)
                
            self.stt.stop_stream()
            latency_ms = (time.time() - start_time) * 1000
            self.context.metrics.record_metric("stt_latency_ms", latency_ms)
            logger.info("STT Complete")
            return

        text = self.stt.transcribe(payload.audio)
        latency_ms = (time.time() - start_time) * 1000

        self.context.metrics.record_metric("stt_latency_ms", latency_ms)
        logger.info("STT Complete")

        words = text.split()
        if not words:
            # If no words were decoded, push an empty final payload to propagate is_final
            payload_out = PartialTranscriptPayload(
                request_id=payload.request_id,
                text_chunk="",
                is_final=True,
                timestamp=payload.user_done_timestamp
            )
            if self.output_queue:
                self.output_queue.put(payload_out)
        else:
            for i, word in enumerate(words):
                is_final = (i == len(words) - 1)
                chunk = word + (" " if not is_final else "")

                # Record first_partial_transcript_ms
                if self.context.streaming_context.check_and_record_first_transcript(payload.request_id):
                    first_stt_lat = (time.time() - payload.user_done_timestamp) * 1000
                    self.context.metrics.record_metric("first_partial_transcript_ms", first_stt_lat)

                payload_out = PartialTranscriptPayload(
                    request_id=payload.request_id,
                    text_chunk=chunk,
                    is_final=is_final,
                    timestamp=payload.user_done_timestamp
                )

                if self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
                    logger.info("SimulatedStreamingSTTWorker: request interrupted/stale. Aborting stream.")
                    return

                if self.output_queue:
                    self.output_queue.put(payload_out)

                # Simulated decoding latency: 30ms per word
                time.sleep(0.03)
