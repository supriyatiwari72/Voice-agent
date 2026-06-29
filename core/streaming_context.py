import logging
import threading

logger = logging.getLogger(__name__)

class StreamingContext:
    """
    Registry tracking first-chunk latencies per request ID to coordinate
    thread-safe telemetry recording in a streaming voice conversational pipeline.
    """

    def __init__(self):
        """
        Initializes the first-chunk telemetry tracking sets.
        """
        self._lock = threading.Lock()
        self._first_transcript_recorded = set()
        self._first_llm_token_recorded = set()
        self._first_sentence_recorded = set()
        self._first_audio_chunk_recorded = set()

    def check_and_record_first_transcript(self, request_id: str) -> bool:
        """
        Thread-safely checks and marks the first partial transcript segment as recorded.
        Returns True if this is the first segment recorded for the request ID.
        """
        with self._lock:
            if request_id not in self._first_transcript_recorded:
                self._first_transcript_recorded.add(request_id)
                return True
            return False

    def check_and_record_first_llm_token(self, request_id: str) -> bool:
        """
        Thread-safely checks and marks the first LLM token as recorded.
        Returns True if this is the first token recorded for the request ID.
        """
        with self._lock:
            if request_id not in self._first_llm_token_recorded:
                self._first_llm_token_recorded.add(request_id)
                return True
            return False

    def check_and_record_first_sentence(self, request_id: str) -> bool:
        """
        Thread-safely checks and marks the first aggregated sentence chunk as recorded.
        Returns True if this is the first sentence completed for the request ID.
        """
        with self._lock:
            if request_id not in self._first_sentence_recorded:
                self._first_sentence_recorded.add(request_id)
                return True
            return False

    def check_and_record_first_audio_chunk(self, request_id: str) -> bool:
        """
        Thread-safely checks and marks the first playback audio chunk as played/recorded.
        Returns True if this is the first chunk received for the request ID.
        """
        with self._lock:
            if request_id not in self._first_audio_chunk_recorded:
                self._first_audio_chunk_recorded.add(request_id)
                return True
            return False

    def clear_request(self, request_id: str) -> None:
        """
        Removes all flags for a completed or interrupted request ID.
        """
        with self._lock:
            self._first_transcript_recorded.discard(request_id)
            self._first_llm_token_recorded.discard(request_id)
            self._first_sentence_recorded.discard(request_id)
            self._first_audio_chunk_recorded.discard(request_id)
