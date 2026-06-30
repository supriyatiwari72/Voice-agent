import logging
from typing import Any

logger = logging.getLogger(__name__)

class PlaybackController:
    """
    Centralized controller for Friday's audio playback.
    Decouples workers from directly managing the speaker audio device and queues.
    """
    def __init__(self, context: Any, player: Any, queue_manager: Any):
        self.context = context
        self.player = player
        self.queue_manager = queue_manager

    def play(self, audio_chunk: bytes) -> None:
        """Pushes synthesized audio chunk to speaker buffer."""
        if self.player and self.player.output_buffer:
            self.player.output_buffer.push(audio_chunk)

    def pause(self) -> None:
        """Pauses the playback stream."""
        logger.info("PlaybackController: pause stream.")

    def resume(self) -> None:
        """Resumes the playback stream."""
        logger.info("PlaybackController: resume stream.")

    def stop(self) -> None:
        """Stops the speaker playback immediately (e.g. for barge-in interruption)."""
        if self.player:
            self.player.interrupt()

    def cancel(self, request_id: str) -> None:
        """Cancels a specific request ID and stops playback."""
        logger.info(f"PlaybackController: cancel request {request_id}")
        if self.context:
            self.context.cancel_request(request_id)
        self.stop()

    def flush(self) -> None:
        """Flushes the playback queues and clears player buffers."""
        logger.info("PlaybackController: flushing playback queues.")
        self.stop()
        if self.queue_manager:
            self.queue_manager.flush_queue("partial_audio_queue")
            self.queue_manager.flush_queue("playback_queue")
