import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import PartialResponsePayload, SentencePayload

logger = logging.getLogger(__name__)

class SentenceAggregatorWorker(BaseWorker):
    """
    Worker that consumes PartialResponsePayload tokens, builds semantic sentences
    based on punctuation boundaries (.?!,;:), and dispatches SentencePayload elements
    to the tts_queue without waiting for the full stream completion.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any):
        """
        Initializes the SentenceAggregatorWorker.
        """
        super().__init__(name="SentenceAggregatorWorker", context=context, input_queue=input_queue, output_queue=output_queue)
        self._buffer = ""

    def process(self, payload: PartialResponsePayload) -> None:
        """
        Process a single PartialResponsePayload token chunk.
        """
        if not payload or not isinstance(payload, PartialResponsePayload):
            logger.warning("Received invalid or empty payload in SentenceAggregatorWorker.")
            return

        # Verify request is active and not interrupted
        if self.context.interruption_event.is_set() or payload.request_id != self.context.get_active_request_id():
            logger.info(f"SentenceAggregatorWorker: request {payload.request_id} is stale/interrupted. Dropping.")
            self._buffer = ""
            return

        # Add incoming token chunk to buffer
        self._buffer += payload.token_chunk

        # Process and yield complete sentences based on boundary marks (. ? ! , ; :)
        while True:
            boundary_idx = -1
            for i, char in enumerate(self._buffer):
                if char in ('.', '?', '!', ',', ';', ':'):
                    boundary_idx = i
                    break

            if boundary_idx == -1:
                break

            # Slice the sentence including punctuation
            sentence = self._buffer[:boundary_idx + 1].strip()
            self._buffer = self._buffer[boundary_idx + 1:]

            if sentence:
                self._dispatch_sentence(payload.request_id, sentence, is_final=False, timestamp=payload.timestamp)

        # If this is the final token payload of the turn, flush any remaining text
        if payload.is_final:
            final_sentence = self._buffer.strip()
            self._buffer = ""
            if final_sentence:
                self._dispatch_sentence(payload.request_id, final_sentence, is_final=True, timestamp=payload.timestamp)
            else:
                # Propagate final turn signal
                self._dispatch_sentence(payload.request_id, "", is_final=True, timestamp=payload.timestamp)

    def _dispatch_sentence(self, request_id: str, text: str, is_final: bool, timestamp: float) -> None:
        """
        Helper method that creates a SentencePayload and puts it on output queue.
        """
        # Record first_sentence_ms telemetry
        if text.strip() and self.context.streaming_context.check_and_record_first_sentence(request_id):
            first_sent_ms = (time.time() - timestamp) * 1000
            self.context.metrics.record_metric("first_sentence_ms", first_sent_ms)

        sentence_payload = SentencePayload(
            request_id=request_id,
            text=text,
            is_final=is_final,
            user_done_timestamp=timestamp
        )

        if self.output_queue:
            self.output_queue.put(sentence_payload)
            logger.debug(f"SentenceAggregatorWorker dispatched sentence: '{text}' (is_final={is_final})")

    def handle_error(self, e: Exception) -> None:
        """
        Resets active text buffers on error.
        """
        self._buffer = ""
