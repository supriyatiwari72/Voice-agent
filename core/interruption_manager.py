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
        Processes an interruption event, halts audio playback, flushes intermediate queues,
        sets the pipeline state back to LISTENING, and triggers event hooks.
        """
        logger.info(f"Handling interruption for request: {request_id}")
        
        # 1. Transition state to INTERRUPTED
        self.context.set_state(PipelineState.INTERRUPTED)
        
        # 2. Trigger INTERRUPTION_STARTED
        self.context.trigger_event(EventType.INTERRUPTION_STARTED, request_id)

        # 3. Trigger interrupt callbacks to stop playback immediately
        self.context.trigger_interrupt_callbacks()

        # 4. Flush intermediate communication queues
        qm = self.context.queue_manager
        qm.flush_queue("transcript_queue")
        qm.flush_queue("partial_transcript_queue")
        qm.flush_queue("partial_response_queue")
        qm.flush_queue("tts_queue")
        qm.flush_queue("partial_audio_queue")
        qm.flush_queue("playback_queue")

        # 5. Clear interruption event so the new speech turn can proceed
        self.context.interruption_event.clear()

        # 6. Set state to LISTENING
        self.context.set_state(PipelineState.LISTENING)

        # 7. Trigger INTERRUPTION_FINISHED
        self.context.trigger_event(EventType.INTERRUPTION_FINISHED, request_id)

        logger.info(f"Interruption handling completed for request {request_id}. Pipeline state is now LISTENING.")
