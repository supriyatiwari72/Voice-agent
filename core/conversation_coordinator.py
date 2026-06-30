import logging
import uuid
from typing import Any
from pipeline.pipeline_state import PipelineState, TurnState
from core.events import (
    SpeechStartedEvent, SpeechEndedEvent, PlaybackCompletedEvent,
    LLMStartedEvent, TTSStartedEvent
)

logger = logging.getLogger(__name__)

class ConversationCoordinator:
    """
    Centralized conversation turns orchestrator (Single Source of Truth).
    Translates typed events from the Event Bus into state changes.
    """
    def __init__(self, context: Any):
        self.context = context
        context.conversation_coordinator = self
        self.active_turn_id = None
        self.active_turn_state = TurnState.CREATED

    def on_event(self, event: Any) -> None:
        """Central event dispatcher for the Event Bus."""
        if isinstance(event, SpeechStartedEvent):
            self.handle_speech_started(event)
        elif isinstance(event, SpeechEndedEvent):
            self.handle_speech_ended(event)
        elif isinstance(event, LLMStartedEvent):
            self.handle_llm_started(event)
        elif isinstance(event, TTSStartedEvent):
            self.handle_tts_started(event)
        elif isinstance(event, PlaybackCompletedEvent):
            self.handle_playback_completed(event)

    def handle_speech_started(self, event: SpeechStartedEvent) -> None:
        """Fired when VAD detects speech onset."""
        current_state = self.context.get_state()
        
        # If Friday is speaking, this is a barge-in interruption
        if current_state == PipelineState.SPEAKING:
            logger.info("Barge-in detected in SPEAKING state. Triggering interruption sequence.")
            self.context.set_state(PipelineState.INTERRUPTED)
            self.active_turn_state = TurnState.CANCELLED
            self.context.interruption_event.set()
            self.context.barge_in_occurred.set()
            
            # Cancel the active request immediately
            active_req = self.context.get_active_request_id()
            if active_req:
                if hasattr(self.context, "interruption_manager") and self.context.interruption_manager:
                    self.context.interruption_manager.handle_interruption(active_req)
                else:
                    self.context.cancel_request(active_req)
                    if self.context.playback_controller:
                        self.context.playback_controller.stop()

        # Create a new conversation turn and transition state to LISTENING
        self.active_turn_id = f"turn-{uuid.uuid4()}"
        self.active_turn_state = TurnState.LISTENING
        self.context.set_active_request_id(event.request_id)
        self.context.set_state(PipelineState.LISTENING)
        logger.info(f"ConversationCoordinator: new turn {self.active_turn_id} created for request {event.request_id}.")

    def handle_speech_ended(self, event: SpeechEndedEvent) -> None:
        """Fired when VAD detects silence timeout."""
        self.active_turn_state = TurnState.TRANSCRIBING
        self.context.set_state(PipelineState.PROCESSING)

    def handle_llm_started(self, event: LLMStartedEvent) -> None:
        """Fired when LLM begins token generation."""
        self.active_turn_state = TurnState.THINKING
        self.context.set_state(PipelineState.THINKING)

    def handle_tts_started(self, event: TTSStartedEvent) -> None:
        """Fired when TTS starts synthesizing voice."""
        self.active_turn_state = TurnState.SPEAKING
        self.context.set_state(PipelineState.SPEAKING)

    def handle_playback_completed(self, event: PlaybackCompletedEvent) -> None:
        """Fired when playback worker finishes final chunk of turn reply."""
        active_req = self.context.get_active_request_id()
        if event.request_id == active_req:
            self.active_turn_state = TurnState.COMPLETED
            self.context.set_state(PipelineState.IDLE)
            self.context.remove_cancellation_token(event.request_id)
            logger.info(f"ConversationCoordinator: turn completed. request {event.request_id} cleared.")
