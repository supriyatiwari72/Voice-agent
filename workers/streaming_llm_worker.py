import logging
import time
from typing import Any, Dict, Optional
from core.worker_base import BaseWorker
from core.payloads import PartialTranscriptPayload, PartialResponsePayload
from pipeline.pipeline_state import PipelineState
from core.interruption_detector import InterruptionDetector, is_goodbye

logger = logging.getLogger(__name__)


class StreamingLLMWorker(BaseWorker):
    """
    Worker that retrieves PartialTranscriptPayload segments, aggregates them,
    and runs streaming token generation when the final segment is received,
    delivering PartialResponsePayload packets.

    Supports semantic interruption detection: if the user says "wait", "stop",
    or "interrupt" during assistant speech, the LLM stream is paused, context
    is saved, and the interruption is acknowledged before resuming.
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

        # Verify request is active and not interrupted
        if self.stop_event.is_set() or self.context.interruption_event.is_set() or \
                payload.request_id != self.context.get_active_request_id():
            logger.info(
                f"StreamingLLMWorker: request {payload.request_id} is stale/interrupted/stopped. "
                f"active={self.context.get_active_request_id()} Dropping."
            )
            self._accumulated_transcripts.pop(payload.request_id, None)
            return

        # Accumulate text chunks (streaming STT path sends multiple partials)
        self._accumulated_transcripts[payload.request_id] = (
            self._accumulated_transcripts.get(payload.request_id, "")
            + payload.text_chunk
        )

        # Wait until the final segment signals the full transcript is ready
        if not payload.is_final:
            # Prefetch / construct conversational history in background early
            prompt = self._accumulated_transcripts[payload.request_id]
            if hasattr(self.context, "memory_manager") and self.context.memory_manager:
                self._prefetch_prompt = self.context.memory_manager.get_context(prompt)
            else:
                self._prefetch_prompt = prompt
            return

        full_text = self._accumulated_transcripts.pop(payload.request_id, "").strip()
        if not full_text:
            full_text = "..."

        # ── Check for goodbye → signal conversation end ────────────────
        if is_goodbye(full_text):
            logger.info("Goodbye detected. Signaling conversation end.")
            if hasattr(self.context, "conversation_recorder") and self.context.conversation_recorder:
                self.context.conversation_recorder.close()
            self.context.shutdown_event.set()
            return

        # ── Check if this is a follow-up from an interruption context ──
        if self._pending_context is not None:
            combined = f"{self._pending_context}\n\n[User adds]: {full_text}"
            full_text = combined
            self._pending_context = None
            logger.info("Resuming LLM with interruption context")

        # Record speech_end → LLM start latency
        speech_end_ms = (time.time() - payload.timestamp) * 1000
        self.context.metrics.record_metric("speech_end_ms", speech_end_ms)

        # Transit to THINKING and then GENERATING
        self.context.set_state(PipelineState.THINKING)
        self.context.set_state(PipelineState.GENERATING)
        logger.info(f"LLM Request Started — prompt length={len(full_text)} chars, request_id={payload.request_id}")

        # ── Memory-augmented prompt ──────────────────────────────────────
        has_memory = hasattr(self.context, "memory_manager") and self.context.memory_manager
        if has_memory:
            self.context.memory_manager.add_user_message(full_text)
            prompt = self.context.memory_manager.get_context(full_text)
        else:
            prompt = full_text

        start_time = time.time()
        first_token = True
        full_response = ""

        # ── Live terminal streaming header ───────────────────────────────
        print("\n[Friday]\n", end="", flush=True)

        # ── Token stream ─────────────────────────────────────────────────
        stream = self.llm.generate_stream(prompt)
        for token in stream:
            # Mid-stream interruption / stop check
            if self.stop_event.is_set() or self.context.interruption_event.is_set() or \
                    payload.request_id != self.context.get_active_request_id():
                logger.info(
                    f"StreamingLLMWorker: request {payload.request_id} interrupted/stopped mid-stream. Aborting."
                )
                print()  # newline to clean up terminal
                return

            full_response += token

            # Live terminal display
            print(token, end="", flush=True)

            if first_token:
                ttft_ms = (time.time() - start_time) * 1000
                self.context.metrics.record_metric("ttft_ms", ttft_ms)

                if self.context.streaming_context.check_and_record_first_llm_token(payload.request_id):
                    first_llm_ms = (time.time() - payload.timestamp) * 1000
                    self.context.metrics.record_metric("first_llm_token_ms", first_llm_ms)

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

        # Newline after streamed response
        print(flush=True)

        # Final interruption / stop check after stream completes
        if self.stop_event.is_set() or self.context.interruption_event.is_set() or \
                payload.request_id != self.context.get_active_request_id():
            logger.info(
                f"StreamingLLMWorker: request {payload.request_id} interrupted/stopped after stream. Discarding."
            )
            return

        latency_ms = (time.time() - start_time) * 1000
        self.context.metrics.record_metric("llm_latency_ms", latency_ms)
        logger.info(f"LLM Complete — total latency={latency_ms:.0f} ms, tokens≈{len(full_response.split())}")
        logger.info("LLM Complete")

        # Store assistant turn → async background summarization
        if has_memory:
            self.context.memory_manager.add_assistant_message(full_response)

        self._previous_response = full_response

        # ── Check if LLM response itself signals conversation end ──────
        if is_goodbye(full_response):
            logger.info("LLM response contains goodbye. Signaling conversation end.")
            self.context.shutdown_event.set()
            return

        # ── Check follow-up 'no' → end conversation ─────────────────────
        if payload.request_id.startswith("followup-"):
            import re
            if re.search(r"\b(no|nope|nah|not really|i.m done|that.s all)\b", full_text, re.IGNORECASE):
                logger.info("User declined follow-up. Ending conversation.")
                self.context.shutdown_event.set()
                return

        # Push closing payload with is_final=True to flush SentenceAggregator
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

    def save_interruption_context(self, partial_response: str):
        """
        Called when a semantic interruption is detected mid-stream.
        Saves the partial response so the next turn can incorporate it.
        """
        if partial_response:
            self._pending_context = (
                f"[Previous assistant response (interrupted)]: {partial_response}"
            )
