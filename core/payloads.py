from dataclasses import dataclass

@dataclass
class AudioPayload:
    """
    Payload containing raw audio bytes from the input device.
    """
    request_id: str
    audio: bytes
    created_at: float

@dataclass
class SpeechPayload:
    """
    Payload containing a complete segment of speech bytes.
    """
    request_id: str
    audio: bytes
    user_done_timestamp: float

@dataclass
class TranscriptPayload:
    """
    Payload containing transcribed user speech text.
    """
    request_id: str
    text: str
    user_done_timestamp: float

@dataclass
class ResponsePayload:
    """
    Payload containing the LLM generation response text.
    """
    request_id: str
    response: str
    user_done_timestamp: float

@dataclass
class TTSPayload:
    """
    Payload containing synthesized text-to-speech audio bytes.
    """
    request_id: str
    audio: bytes
    user_done_timestamp: float

@dataclass
class InterruptionPayload:
    """
    Payload containing the request ID being interrupted and the timestamp.
    """
    request_id: str
    timestamp: float

@dataclass
class PartialTranscriptPayload:
    """
    Payload containing a partial decoded STT transcript text segment.
    """
    request_id: str
    text_chunk: str
    is_final: bool
    timestamp: float

@dataclass
class PartialResponsePayload:
    """
    Payload containing a partial LLM generated token segment.
    """
    request_id: str
    token_chunk: str
    is_final: bool
    timestamp: float

@dataclass
class PartialAudioPayload:
    """
    Payload containing a partial synthesized text-to-speech audio segment.
    """
    request_id: str
    audio_chunk: bytes
    is_final: bool
    timestamp: float

@dataclass
class SentencePayload:
    """
    Payload containing a consolidated semantic sentence chunk for TTS.
    """
    request_id: str
    text: str
    is_final: bool
    user_done_timestamp: float

@dataclass
class StreamControlPayload:
    """
    Payload containing explicit stream lifecycle control tracking.
    """
    request_id: str
    stream_complete: bool
    interruption: bool


@dataclass
class StreamingAudioPayload:
    """
    Payload containing a raw audio chunk from microphone, streamed in real-time.
    """
    request_id: str
    audio_chunk: bytes
    is_final: bool
    timestamp: float



