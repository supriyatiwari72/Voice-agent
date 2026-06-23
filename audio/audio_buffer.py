import queue
from typing import Optional

class AudioBuffer:
    """
    A thread-safe double-ended queue or circular buffer for transfering raw audio bytes
    between input/output hardware threads and the processing pipeline thread.
    """

    def __init__(self, max_size: int = 100):
        """
        Initializes the thread-safe Queue buffer.
        """
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)

    def push(self, chunk: bytes, timeout: Optional[float] = None) -> bool:
        """
        Pushes a new audio frame segment (bytes) into the buffer.

        Args:
            chunk (bytes): Raw audio frames.
            timeout (Optional[float]): Block duration timeout.

        Returns:
            bool: True if pushed successfully, False if timeout or buffer is full.
        """
        try:
            self._queue.put(chunk, block=True, timeout=timeout)
            return True
        except queue.Full:
            return False

    def pop(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Pops the oldest audio frame segment (bytes) from the buffer.

        Args:
            timeout (Optional[float]): Block duration timeout.

        Returns:
            Optional[bytes]: Raw audio chunk, or None if buffer is empty or timeout.
        """
        try:
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """
        Flushes all chunks from the buffer instantly.
        """
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.all_tasks_done.notify_all()
            self._queue.unfinished_tasks = 0

    def size(self) -> int:
        """
        Returns the current number of frames inside the buffer.
        """
        return self._queue.qsize()
