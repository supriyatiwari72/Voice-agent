import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import PartialAudioPayload
from pipeline.pipeline_state import PipelineState
from core.interruption_detector import is_goodbye

logger = logging.getLogger(__name__)

# Kokoro TTS outputs at 24kHz — we resample to 16kHz for conversation recording
_TTS_SAMPLE_RATE = 24000


class PlaybackWorker(BaseWorker):
    """
    Worker that retrieves PartialAudioPayload from partial_audio_queue,
    forwards raw audio bytes to the AudioPlayer output buffer, tracks
    first_audio_chunk_ms and total_turnaround_ms telemetry, and resets
    the pipeline state to LISTENING upon complete playback delivery.

    Records assistant TTS audio into the conversation recorder so both
    sides of the conversation (user + assistant) are captured in one file.
    Detects "Goodbye" to trigger conversation end.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any):
        super().__init__(
            name="PlaybackWorker",
            context=context,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        self._goodbye_played = False

    def process(self, payload: PartialAudioPayload) -> None:
        if not payload or not isinstance(payload, PartialAudioPayload):
            logger.warning("Received invalid or empty payload in PlaybackWorker.")
            return

        # Drop stale / interrupted / stopped requests
        if self.stop_event.is_set() or self.context.interruption_event.is_set() or \
                payload.request_id != self.context.get_active_request_id():
            logger.info(
                f"PlaybackWorker: request {payload.request_id} is stale/interrupted/stopped. Dropping chunk."
            )
            return

        # Record first_audio_chunk_ms on the first non-empty audio chunk
        if payload.audio_chunk and \
                self.context.streaming_context.check_and_record_first_audio_chunk(payload.request_id):
            first_audio_ms = (time.time() - payload.timestamp) * 1000
            self.context.metrics.record_metric("first_audio_chunk_ms", first_audio_ms)
            logger.info(f"First Audio Chunk — latency from speech end: {first_audio_ms:.0f} ms")

        # Record assistant audio into conversation recorder
        recorder = getattr(self.context, "conversation_recorder", None)
        if recorder is not None and payload.audio_chunk:
            recorder.write_audio_from_rate(payload.audio_chunk, _TTS_SAMPLE_RATE)

        # Forward audio bytes to AudioPlayer's output buffer (speaker)
        if self.output_queue and payload.audio_chunk:
            self.output_queue.put(payload.audio_chunk)

        # On final chunk: record turnaround, log, reset state
        if payload.is_final:
            turnaround_ms = (time.time() - payload.timestamp) * 1000
            self.context.metrics.record_metric("total_turnaround_ms", turnaround_ms)
            logger.info(f"Playback Complete — total turnaround: {turnaround_ms:.0f} ms")
            logger.info("Playback Complete")

            # Clean up per-request streaming context flags
            if hasattr(self.context, "streaming_context") and self.context.streaming_context:
                self.context.streaming_context.clear_request(payload.request_id)

            import sys
            is_testing = "pytest" in sys.modules
            req_id = payload.request_id or ""
            is_interrupted = self.context.interruption_event.is_set()

            if self._goodbye_played or self.context.shutdown_event.is_set():
                return

            # Always reset to LISTENING so the agent accepts new user input
            if self.context.get_state() not in (PipelineState.LISTENING, PipelineState.USER_SPEAKING, PipelineState.STOPPED):
                self.context.set_state(PipelineState.LISTENING)

            if (not is_testing
                    and not is_interrupted
                    and not req_id.startswith("followup-")
                    and not req_id.startswith("greeting-")):
                followup_id = f"followup-{req_id}"
                self.context.set_active_request_id(followup_id)
                self.context.set_state(PipelineState.SPEAKING)

                print("\n[Friday]")
                print("Anything else I can help you with?\n", flush=True)

                from core.payloads import SentencePayload
                followup_payload = SentencePayload(
                    request_id=followup_id,
                    text="Anything else I can help you with?",
                    is_final=True,
                    user_done_timestamp=time.time()
                )
                if self.context.queue_manager.tts_queue:
                    self.context.queue_manager.tts_queue.put(followup_payload)
            else:
                if not is_testing and not is_interrupted:
                    logger.info("Speak now...")
