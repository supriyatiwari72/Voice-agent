from enum import Enum, auto

class EventType(Enum):
    """
    Enum representing system-wide coordination and lifecycle events.
    """
    STARTUP_EVENT = auto()
    SHUTDOWN_EVENT = auto()
    ERROR_EVENT = auto()

    # Speech lifecycle events (fired by VADWorker)
    SPEECH_STARTED = auto()     # User has begun speaking
    SPEECH_ENDED = auto()       # User has finished speaking (silence detected)

    # Interruption and Barge-in (Phase 3C)
    INTERRUPTION_STARTED = auto()
    INTERRUPTION_FINISHED = auto()
