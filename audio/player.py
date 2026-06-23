import logging
import threading
import time
from typing import Dict, Any
from audio.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)

class AudioPlayer:
    """
    Simulates hardware playback player for Phase 1. Consumes audio bytes and logs completions.
    """

    def __init__(self, config: Dict[str, Any], output_buffer: AudioBuffer):
        self.config = config
        self.output_buffer = output_buffer
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def start_playback(self) -> None:
        if self._active:
            return
        
        self._active = True
        self._thread = threading.Thread(target=self._simulate_playback, daemon=True)
        self._thread.start()

    def _simulate_playback(self) -> None:
        while self._active:
            chunk = self.output_buffer.pop(timeout=0.1)
            if chunk:
                # Simulate the time it takes to play back the audio (e.g., brief pause)
                time.sleep(0.05)
                print("Playback Complete")

    def stop_playback(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def interrupt(self) -> None:
        self.output_buffer.clear()

    def is_active(self) -> bool:
        return self._active
