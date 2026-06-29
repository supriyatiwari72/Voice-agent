from abc import ABC, abstractmethod

class BaseSTT(ABC):
    """
    Abstract base class establishing the contract for Speech-To-Text (STT) transcription providers.
    """

    def supports_streaming_audio(self) -> bool:
        """Returns True if this provider can process streaming raw audio chunks."""
        return False

    def supports_partial_transcripts(self) -> bool:
        """Returns True if this provider can emit partial transcripts during a turn."""
        return False

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe raw audio frame segments into a text string.

        Args:
            audio_data (bytes): Combined raw audio bytes representing a complete speech turn.

        Returns:
            str: Transcribed text output.
        """
        pass

class BaseStreamingSTT(ABC):
    """
    Abstract base class establishing the contract for streaming Speech-To-Text (STT) providers.
    """

    def supports_streaming_audio(self) -> bool:
        return True

    def supports_partial_transcripts(self) -> bool:
        return True

    @abstractmethod
    def start_stream(self, request_id: str, on_transcript_cb) -> None:
        """
        Starts the streaming session.

        Args:
            request_id (str): The active request identifier.
            on_transcript_cb (Callable[[str, bool], None]): Callback triggered with (chunk_text, is_final).
        """
        pass

    @abstractmethod
    def stream_audio(self, audio_chunk: bytes) -> None:
        """
        Streams a chunk of audio to the provider.

        Args:
            audio_chunk (bytes): Raw audio frames.
        """
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """
        Stops the streaming session.
        """
        pass
