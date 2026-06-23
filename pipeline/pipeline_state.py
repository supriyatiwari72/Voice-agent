from enum import Enum, auto

class PipelineState(Enum):
    """
    Represents the operational states of the Voice-to-Voice AI Agent pipeline.
    Useful for orchestrating asynchronous loops, rendering UI feedback,
    and managing speech interruption events.
    """
    IDLE = auto()          # Pipeline is active but waiting for microphone streams or triggers.
    LISTENING = auto()     # Capturing raw microphone audio frames into memory buffers.
    PROCESSING = auto()    # Pre-processing frame segments (noise suppression, VAD analysis).
    TRANSCRIBING = auto()  # Converting filtered speech frames into text using Speech-to-Text.
    THINKING = auto()      # Submitting transcripts to the LLM and awaiting response tokens.
    SPEAKING = auto()      # Synthesizing and playing back audio chunks via the text-to-speech engine.
    ERROR = auto()         # An exception occurred; logging diagnostics and cleaning up buffers.
    STOPPED = auto()       # The pipeline is shut down and all audio hardware hooks are released.
