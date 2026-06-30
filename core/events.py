from enum import Enum, auto
from dataclasses import dataclass

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


# ─── TYPED EVENT CLASSES ──────────────────────────────────────────────────────────

@dataclass
class SpeechStartedEvent:
    request_id: str
    timestamp: float

@dataclass
class SpeechEndedEvent:
    request_id: str
    timestamp: float

@dataclass
class PartialTranscriptEvent:
    request_id: str
    text_chunk: str
    timestamp: float

@dataclass
class FinalTranscriptEvent:
    request_id: str
    text: str
    timestamp: float

@dataclass
class LLMStartedEvent:
    request_id: str
    timestamp: float

@dataclass
class LLMFirstTokenEvent:
    request_id: str
    token: str
    timestamp: float

@dataclass
class LLMCompletedEvent:
    request_id: str
    full_response: str
    timestamp: float

@dataclass
class TTSStartedEvent:
    request_id: str
    timestamp: float

@dataclass
class PlaybackStartedEvent:
    request_id: str
    timestamp: float

@dataclass
class PlaybackCompletedEvent:
    request_id: str
    timestamp: float

@dataclass
class RequestCancelledEvent:
    request_id: str
    timestamp: float

@dataclass
class MicrophoneFailureEvent:
    error_message: str
    timestamp: float
