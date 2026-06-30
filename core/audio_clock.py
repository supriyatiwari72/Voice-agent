import time
import threading

class AudioClock:
    """
    A shared audio clock to track elapsed audio samples and wall time
    across capture, VAD, and playback stages.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._samples_processed = 0
        self._start_time = time.time()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Resets the clock parameters."""
        with self._lock:
            self._samples_processed = 0
            self._start_time = time.time()

    def advance(self, num_samples: int) -> None:
        """Advances the clock by processing a number of samples."""
        with self._lock:
            self._samples_processed += num_samples

    def get_time(self) -> float:
        """Returns the elapsed wall clock time in seconds."""
        with self._lock:
            return time.time() - self._start_time

    def get_audio_time(self) -> float:
        """Returns the processed audio duration in seconds."""
        with self._lock:
            return self._samples_processed / self.sample_rate

    def get_samples(self) -> int:
        """Returns total samples processed."""
        with self._lock:
            return self._samples_processed
