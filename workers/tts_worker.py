import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import ResponsePayload, TTSPayload
from pipeline.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class TTSWorker(BaseWorker):
    """
    Worker that retrieves ResponsePayload, runs text-to-speech synthesis,
    records TTS latency, and forwards TTSPayload to playback_queue.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, tts: Any):
        """
        Initializes the TTSWorker.
        """
        super().__init__(name="TTSWorker", context=context, input_queue=input_queue, output_queue=output_queue)
        self.tts = tts

    def process(self, payload: ResponsePayload) -> None:
        """
        Process a single ResponsePayload turn.
        """
        if not payload or not isinstance(payload, ResponsePayload):
            logger.warning("Received invalid or empty payload in TTSWorker.")
            return

        # Verify request is active before calling TTS
        if self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
            logger.info(f"TTSWorker: request {payload.request_id} is stale/interrupted. Dropping payload before synthesis.")
            return

        # Transit pipeline state
        self.context.set_state(PipelineState.SPEAKING)

        start_time = time.time()
        audio_bytes = self.tts.synthesize(payload.response)
        
        # Verify request is active after synthesis is done
        if self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
            logger.info(f"TTSWorker: request {payload.request_id} is stale/interrupted. Discarding synthesis output.")
            return

        latency_ms = (time.time() - start_time) * 1000

        # Record metric logs
        self.context.metrics.record_metric("tts_latency_ms", latency_ms)
        logger.info("TTS Complete")  # Print matching validation expectation

        tts_payload = TTSPayload(
            request_id=payload.request_id,
            audio=audio_bytes,
            user_done_timestamp=payload.user_done_timestamp
        )

        if self.output_queue:
            self.output_queue.put(tts_payload)

