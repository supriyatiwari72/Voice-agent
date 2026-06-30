from enum import Enum, auto

class PipelineState(Enum):
    """
    Represents the global operational states of the Voice-to-Voice AI Agent pipeline.
    """
    INITIALIZING = auto()
    READY = auto()
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()
    SHUTDOWN = auto()


class TurnState(Enum):
    """
    Represents the lifecycle states of a single conversation turn.
    """
    CREATED = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()
