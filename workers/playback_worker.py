import logging
import time
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import PartialAudioPayload
from pipeline.pipeline_state import PipelineState
from core.events import EventType, PlaybackCompletedEvent

logger = logging.getLogger(__name__)

# Kokoro TTS outputs at 24kHz — resampled to 16kHz for conversation recording
_TTS_SAMPLE_RATE = 24000


class PlaybackWorker(BaseWorker):
    """
    Worker that retrieves PartialAudioPayload from partial_audio_queue,
    forwards raw audio bytes to the AudioPlayer output buffer via PlaybackController,
    tracks telemetry, and resets the pipeline state to IDLE upon complete playback delivery.
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

        # Drop stale / cancelled requests
        if self.stop_event.is_set() or self.context.is_request_cancelled(payload.request_id) or \
                payload.request_id != self.context.get_active_request_id():
            logger.info(
                f"PlaybackWorker: request {payload.request_id} is cancelled/stale/stopped. Dropping chunk."
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

        # Forward audio bytes to AudioPlayer's output buffer (speaker) via PlaybackController
        if payload.audio_chunk:
            if self.context.playback_controller:
                self.context.playback_controller.play(payload.audio_chunk)
            elif self.output_queue:
                self.output_queue.put(payload.audio_chunk)

        # On final chunk: record turnaround, log, reset state to IDLE
        if payload.is_final:
            # Wait for physical playback to complete (buffer to drain)
            if self.context and self.context.playback_controller and self.context.playback_controller.player:
                player = self.context.playback_controller.player
                if player.is_active():
                    while player.output_buffer.size() > 0:
                        if self.context.is_request_cancelled(payload.request_id):
                            break
                        time.sleep(0.02)
                    # Small extra grace period to let the final chunk play out of speakers
                    if not self.context.is_request_cancelled(payload.request_id):
                        time.sleep(0.15)

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
            is_cancelled = self.context.is_request_cancelled(req_id)

            if self._goodbye_played or self.context.shutdown_event.is_set():
                return

            # Trigger PlaybackCompletedEvent event (Coordinator transitions state to PipelineState.IDLE)
            self.context.trigger_event(
                EventType.ERROR_EVENT,
                PlaybackCompletedEvent(request_id=payload.request_id, timestamp=time.time())
            )

            # Fallback backward-compatible state transition check
            if self.context.get_state() not in (PipelineState.IDLE, PipelineState.LISTENING, PipelineState.SHUTDOWN):
                self.context.set_state(PipelineState.IDLE)

            # If follow-up prompt is explicitly enabled in config
            followup_enabled = self.context.config.get("conversation", {}).get("followup_enabled", True)
            if (not is_testing
                    and not is_cancelled
                    and followup_enabled
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
                if not is_testing and not is_cancelled:
                    logger.info("Speak now...")
