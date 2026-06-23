import logging
import threading
import time
import numpy as np
from typing import Dict, Any, Callable, Optional
from audio.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)

class AudioRecorder:
    """
    Simulates hardware audio recorder.
    Generates a clean 100Hz sine wave on the first frame to safely trigger VAD
    without causing Whisper decoder hangs, followed by silence.
    """

    def __init__(self, config: Dict[str, Any], input_buffer: AudioBuffer):
        self.config = config
        self.input_buffer = input_buffer
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def start_recording(self, callback: Callable[[bytes], None] = None) -> None:
        if self._active:
            return
        
        print("Recorder Started")
        self._active = True
        
        # Start background simulator thread
        self._thread = threading.Thread(target=self._simulate_capture, daemon=True)
        self._thread.start()

    def _simulate_capture(self) -> None:
        frame_count = 0
        while self._active:
            if frame_count == 0:
                # First frame: 100Hz sine wave to trigger VAD cleanly
                # 480 samples at 16kHz = 30ms
                t = np.arange(480) / 16000.0
                sine_wave = np.sin(2 * np.pi * 100.0 * t)
                dummy_frame = (sine_wave * 32767.0).astype(np.int16).tobytes()
            else:
                # Subsequent frames: Silent PCM
                dummy_frame = b"\x00" * 960

            self.input_buffer.push(dummy_frame)
            frame_count += 1
            time.sleep(0.03)

    def stop_recording(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_active(self) -> bool:
        return self._active
