import logging
from typing import Any
from core.events import EventType
from pipeline.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class InterruptionManager:
    """
    Manages interruption and barge-in lifecycles, flushes active queues,
    cancels current player output, and notifies registered listeners.
    """

    def __init__(self, context: Any):
        """
        Initializes the InterruptionManager with a shared PipelineContext.
        """
        self.context = context

    def handle_interruption(self, request_id: str) -> None:
        """
        Processes an interruption event, halts audio playback, flushes downstream
        assistant queues (while leaving STT queues intact to preserve the interruption speech),
        sets the pipeline state back to LISTENING, and triggers event hooks.
        """
        logger.info(f"Handling interruption for request: {request_id}")

        # 1. Transition state to INTERRUPTED
        self.context.set_state(PipelineState.INTERRUPTED)

        # 2. Trigger INTERRUPTION_STARTED
        self.context.trigger_event(EventType.INTERRUPTION_STARTED, request_id)

        # 3. Cancel the current turn's request and stop playback immediately
        if self.context.playback_controller:
            self.context.playback_controller.cancel(request_id)
            self.context.playback_controller.flush()
        else:
            self.context.trigger_interrupt_callbacks()

        # 4. Flush ONLY downstream assistant generation queues.
        #    Do NOT flush speech_queue or transcript_queue/partial_transcript_queue,
        #    as they contain the beginning of the user's interruption speech.
        qm = self.context.queue_manager
        qm.flush_queue("partial_response_queue")
        qm.flush_queue("tts_queue")
        qm.flush_queue("partial_audio_queue")
        qm.flush_queue("playback_queue")

        # 5. Clear the interruption_event flag
        self.context.interruption_event.clear()

        # 6. Set state to LISTENING so the new turn captures speech immediately
        self.context.set_state(PipelineState.LISTENING)

        # 7. Trigger INTERRUPTION_FINISHED
        self.context.trigger_event(EventType.INTERRUPTION_FINISHED, request_id)

        logger.info(f"Interruption handling completed for request {request_id}. Pipeline state is now LISTENING.")
