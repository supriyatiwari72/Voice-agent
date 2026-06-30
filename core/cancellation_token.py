import threading

class CancellationToken:
    """
    A thread-safe cancellation token.
    """
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signals cancellation."""
        with self._lock:
            self._cancelled = True

    def is_cancelled(self) -> bool:
        """Returns True if cancellation has been requested."""
        with self._lock:
            return self._cancelled
