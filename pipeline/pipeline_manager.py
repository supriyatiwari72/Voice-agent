import logging
import threading
import time
from typing import Dict, Any

from audio.audio_buffer import AudioBuffer
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer

from noise.factory import NoiseFactory
from vad.factory import VADFactory
from stt.factory import STTFactory
from llm.factory import LLMFactory
from tts.factory import TTSFactory

from pipeline.voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Coordinates threads, manages queues, and handles lifecycle loops.
    Contains no AI business logic; orchestrates starting/stopping components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline: VoicePipeline = None
        self._is_running = False
        
        # Initialize thread-safe queues
        max_size = config.get("audio", {}).get("buffer_max_size", 100)
        self.input_buffer = AudioBuffer(max_size=max_size)
        self.output_buffer = AudioBuffer(max_size=max_size)
        
        # Audio IO modules
        self.recorder = AudioRecorder(config, self.input_buffer)
        self.player = AudioPlayer(config, self.output_buffer)
        
        self._pipeline_thread: Optional[threading.Thread] = None

    def initialize_pipeline(self) -> None:
        logger.info("Initializing pipeline factories...")
        
        # Resolve active provider names
        providers = self.config.get("active_providers", {})
        
        # Resolve instances using factory Registries
        noise_canceller = NoiseFactory.get_provider(providers.get("noise"), self.config)
        vad = VADFactory.get_provider(providers.get("vad"), self.config)
        stt = STTFactory.get_provider(providers.get("stt"), self.config)
        llm = LLMFactory.get_provider(providers.get("llm"), self.config)
        tts = TTSFactory.get_provider(providers.get("tts"), self.config)
        
        # Construct the pipeline depending only on abstractions
        self.pipeline = VoicePipeline(
            noise_canceller=noise_canceller,
            vad=vad,
            stt=stt,
            llm=llm,
            tts=tts
        )

    def start(self) -> None:
        if self._is_running:
            return

        self._is_running = True
        
        # Start Audio IO
        self.recorder.start_recording()
        self.player.start_playback()
        
        # Start pipeline loop thread
        self._pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._pipeline_thread.start()

    def _pipeline_loop(self) -> None:
        while self._is_running:
            # Pull frame from recorder buffer (non-blocking poll to support graceful stop)
            frame = self.input_buffer.pop(timeout=0.1)
            if frame:
                # Run through the pipeline processing
                output_audio = self.pipeline.process_frame(frame)
                if output_audio:
                    # Push synthesized audio chunks for speaker playback
                    self.output_buffer.push(output_audio)

    def stop(self) -> None:
        if not self._is_running:
            return

        self._is_running = False
        
        # Stop threads
        self.recorder.stop_recording()
        self.player.stop_playback()
        
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=1.0)
            self._pipeline_thread = None

    def is_running(self) -> bool:
        return self._is_running
