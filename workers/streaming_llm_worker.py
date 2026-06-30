import logging
import time
import re
from typing import Any, Dict, Optional
from core.worker_base import BaseWorker
from core.payloads import PartialTranscriptPayload, PartialResponsePayload
from pipeline.pipeline_state import PipelineState
from core.interruption_detector import InterruptionDetector, is_goodbye
from core.events import EventType, LLMStartedEvent, LLMFirstTokenEvent, LLMCompletedEvent

logger = logging.getLogger(__name__)


class StreamingLLMWorker(BaseWorker):
    """
    Worker that retrieves PartialTranscriptPayload segments, aggregates them,
    and runs streaming token generation when the final segment is received,
    delivering PartialResponsePayload packets.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, llm: Any):
        super().__init__(
            name="StreamingLLMWorker",
            context=context,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        self.llm = llm
        self._accumulated_transcripts: Dict[str, str] = {}
        self._interruption_detector = InterruptionDetector()
        self._pending_context: Optional[str] = None
        self._previous_response: Optional[str] = None

    def process(self, payload: PartialTranscriptPayload) -> None:
        if not payload or not isinstance(payload, PartialTranscriptPayload):
            logger.warning("Received invalid or empty payload in StreamingLLMWorker.")
            return

        # Verify request is active and not cancelled
        if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
            logger.info(
                f"StreamingLLMWorker: request {payload.request_id} is cancelled/stopped. Dropping."
            )
            self._accumulated_transcripts.pop(payload.request_id, None)
            return

        # Accumulate text chunks
        self._accumulated_transcripts[payload.request_id] = (
            self._accumulated_transcripts.get(payload.request_id, "")
            + payload.text_chunk
        )

        if not payload.is_final:
            prompt = self._accumulated_transcripts[payload.request_id]
            if hasattr(self.context, "memory_manager") and self.context.memory_manager:
                self._prefetch_prompt = self.context.memory_manager.get_context(prompt)
            else:
                self._prefetch_prompt = prompt
            return

        full_text = self._accumulated_transcripts.pop(payload.request_id, "").strip()
        if not full_text:
            full_text = "..."

        if is_goodbye(full_text):
            logger.info("Goodbye detected. Signaling conversation end.")
            if hasattr(self.context, "conversation_recorder") and self.context.conversation_recorder:
                self.context.conversation_recorder.close()
            self.context.shutdown_event.set()
            return

        if self._pending_context is not None:
            combined = f"{self._pending_context}\n\n[User adds]: {full_text}"
            full_text = combined
            self._pending_context = None
            logger.info("Resuming LLM with interruption context")

        # Record speech_end → LLM start latency
        speech_end_ms = (time.time() - payload.timestamp) * 1000
        self.context.metrics.record_metric("speech_end_ms", speech_end_ms)

        # Trigger LLMStartedEvent event (Coordinator transitions state to PipelineState.THINKING)
        self.context.trigger_event(
            EventType.ERROR_EVENT, 
            LLMStartedEvent(request_id=payload.request_id, timestamp=time.time())
        )
        logger.info(f"LLM Request Started — prompt length={len(full_text)} chars, request_id={payload.request_id}")

        has_memory = hasattr(self.context, "memory_manager") and self.context.memory_manager
        if has_memory:
            self.context.memory_manager.add_user_message(full_text)
            prompt = self.context.memory_manager.get_context(full_text)
        else:
            prompt = full_text

        start_time = time.time()
        first_token = True
        full_response = ""

        print("\n[Friday]\n", end="", flush=True)

        # ── Token stream ─────────────────────────────────────────────────
        stream = self.llm.generate_stream(prompt)
        for token in stream:
            # Check cancellation token on each iteration
            if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
                logger.info(
                    f"StreamingLLMWorker: request {payload.request_id} cancelled mid-stream. Aborting."
                )
                print()
                return

            full_response += token
            print(token, end="", flush=True)

            if first_token:
                ttft_ms = (time.time() - start_time) * 1000
                self.context.metrics.record_metric("ttft_ms", ttft_ms)

                if self.context.streaming_context.check_and_record_first_llm_token(payload.request_id):
                    first_llm_ms = (time.time() - payload.timestamp) * 1000
                    self.context.metrics.record_metric("first_llm_token_ms", first_llm_ms)

                self.context.trigger_event(
                    EventType.ERROR_EVENT,
                    LLMFirstTokenEvent(request_id=payload.request_id, token=token, timestamp=time.time())
                )
                logger.info(f"First Token Received — TTFT={ttft_ms:.0f} ms")
                first_token = False

            response_payload = PartialResponsePayload(
                request_id=payload.request_id,
                token_chunk=token,
                is_final=False,
                timestamp=payload.timestamp,
            )
            if self.output_queue:
                self.output_queue.put(response_payload)

        print(flush=True)

        if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id):
            logger.info(
                f"StreamingLLMWorker: request {payload.request_id} cancelled after stream. Discarding."
            )
            return

        latency_ms = (time.time() - start_time) * 1000
        self.context.metrics.record_metric("llm_latency_ms", latency_ms)
        logger.info(f"LLM Complete — total latency={latency_ms:.0f} ms")

        # Trigger LLMCompletedEvent
        self.context.trigger_event(
            EventType.ERROR_EVENT,
            LLMCompletedEvent(request_id=payload.request_id, full_response=full_response, timestamp=time.time())
        )

        if has_memory:
            self.context.memory_manager.add_assistant_message(full_response)

        self._previous_response = full_response

        if is_goodbye(full_response):
            logger.info("LLM response contains goodbye. Signaling conversation end.")
            self.context.shutdown_event.set()
            return

        if payload.request_id.startswith("followup-"):
            if re.search(r"\b(no|nope|nah|not really|i.m done|that.s all)\b", full_text, re.IGNORECASE):
                logger.info("User declined follow-up. Ending conversation.")
                self.context.shutdown_event.set()
                return

        # Push final empty response payload to complete turn
        closing_payload = PartialResponsePayload(
            request_id=payload.request_id,
            token_chunk="",
            is_final=True,
            timestamp=payload.timestamp,
        )
        if self.output_queue:
            self.output_queue.put(closing_payload)

    def handle_error(self, e: Exception) -> None:
        self._accumulated_transcripts.clear()
        self._pending_context = None
        self.context.set_state(PipelineState.IDLE)
        # Speak a short apology on coordinator if possible
        logger.error(f"StreamingLLMWorker error: {e}")

    def save_interruption_context(self, partial_response: str):
        if partial_response:
            self._pending_context = (
                f"[Previous assistant response (interrupted)]: {partial_response}"
            )
