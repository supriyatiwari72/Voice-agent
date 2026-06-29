import logging
import os
import threading
import time
import uuid
from typing import Dict, Any, Callable
from pipeline.pipeline_state import PipelineState
from core.events import EventType

logger = logging.getLogger(__name__)

_SESSION_COUNTER_FILE = "audio/recordings/.session_counter"


def _next_session_number() -> int:
    """Read, increment, and persist a session counter for sequential naming."""
    os.makedirs(os.path.dirname(_SESSION_COUNTER_FILE), exist_ok=True)
    num = 1
    try:
        if os.path.exists(_SESSION_COUNTER_FILE):
            with open(_SESSION_COUNTER_FILE, "r") as f:
                num = int(f.read().strip()) + 1
    except (ValueError, OSError):
        num = 1
    try:
        with open(_SESSION_COUNTER_FILE, "w") as f:
            f.write(str(num))
    except OSError:
        pass
    return num


class PipelineContext:
    """
    Context structure holding configuration parameters, bounded queue managers, 
    metrics trackers, session tracking, and centralized event notifications.
    """

    def __init__(self, config: Dict[str, Any], queue_manager: Any, metrics_tracker: Any):
        """
        Initializes the PipelineContext.
        """
        self.config = config or {}
        self.queue_manager = queue_manager
        self.metrics = metrics_tracker
        
        # Session identifiers — sequential numbering (session1, session2, ...)
        self.session_number = _next_session_number()
        self.session_id = f"session{self.session_number}"
        self.startup_time = time.time()
        self.active_request_id = None
        self._active_req_lock = threading.Lock()
        
        # Coordination events
        self.shutdown_event = threading.Event()
        self.interruption_event = threading.Event()
        
        # Thread-safe pipeline state management
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # Thread-safe event callback registry
        self._event_callbacks = {event: [] for event in EventType}
        self._callback_lock = threading.Lock()

        # Thread-safe state change callbacks
        self._state_callbacks = []

        # Interruption callbacks
        self._interrupt_callbacks = []
        self._interrupt_lock = threading.Lock()

    def get_state(self) -> PipelineState:
        """
        Retrieves the current thread-safe pipeline execution state.
        """
        with self._state_lock:
            return self._state

    def register_state_listener(self, callback: Callable[[PipelineState], None]) -> None:
        """
        Registers a callback handler for pipeline state changes.
        """
        with self._state_lock:
            self._state_callbacks.append(callback)

    def set_state(self, state: PipelineState) -> None:
        """
        Safely transition to a new pipeline state, logging the transaction and notifying observers.
        """
        callbacks = []
        with self._state_lock:
            if self._state != state:
                logger.info(f"Pipeline state transition: {self._state.name} -> {state.name}")
                self._state = state
                callbacks = list(self._state_callbacks)
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error executing state listener callback: {e}")

    def register_event_listener(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        Registers a callback handler for a specific event type.
        """
        with self._callback_lock:
            self._event_callbacks[event_type].append(callback)

    def trigger_event(self, event_type: EventType, payload: Any = None) -> None:
        """
        Triggers an event and runs all registered callback listeners.
        """
        logger.info(f"System event triggered: {event_type.name}")
        with self._callback_lock:
            callbacks = list(self._event_callbacks.get(event_type, []))
            
        for callback in callbacks:
            try:
                callback(payload)
            except Exception as e:
                logger.error(f"Error executing callback for event {event_type.name}: {e}")

    def register_interrupt_callback(self, callback: Callable[[], None]) -> None:
        """
        Registers a callback to be triggered immediately when an interruption occurs (e.g. to clear audio player buffer).
        """
        with self._interrupt_lock:
            self._interrupt_callbacks.append(callback)

    def trigger_interrupt_callbacks(self) -> None:
        """
        Triggers all registered interrupt callbacks.
        """
        with self._interrupt_lock:
            callbacks = list(self._interrupt_callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"Error executing interrupt callback: {e}")

    def set_active_request_id(self, request_id: str) -> None:
        """
        Thread-safely sets the active request ID.
        """
        with self._active_req_lock:
            self.active_request_id = request_id

    def get_active_request_id(self) -> Optional[str]:
        """
        Thread-safely retrieves the active request ID.
        """
        with self._active_req_lock:
            return self.active_request_id

