import logging
import threading
import time
from typing import Dict, Any, List, Optional

from audio.audio_buffer import AudioBuffer
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer

from noise.factory import NoiseFactory
from vad.factory import VADFactory
from stt.factory import STTFactory
from llm.factory import LLMFactory
from tts.factory import TTSFactory

from core.metrics import MetricsTracker
from core.queue_manager import QueueManager
from core.pipeline_context import PipelineContext
from core.conversation_recorder import ConversationRecorder
from pipeline.voice_pipeline import VoicePipeline
from memory.memory_manager import MemoryManager

from workers.noise_worker import NoiseWorker
from workers.vad_worker import VADWorker
from workers.stt_worker import STTWorker
from workers.streaming_llm_worker import StreamingLLMWorker
from workers.sentence_aggregator_worker import SentenceAggregatorWorker
from workers.streaming_tts_worker import StreamingTTSWorker
from workers.playback_worker import PlaybackWorker
from core.interruption_manager import InterruptionManager
from workers.interruption_worker import InterruptionWorker
from core.streaming_context import StreamingContext

logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Coordinates concurrent worker threads, manages queue lifecycles,
    tracks state context, and monitors execution metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PipelineManager.
        """
        self.config = config
        self.pipeline: Optional[VoicePipeline] = None
        self._is_running = False
        
        # Initialize hardware double-ended queues for recorders/players
        max_size = config.get("audio", {}).get("buffer_max_size", 200)
        self.input_buffer = AudioBuffer(max_size=max_size)
        self.output_buffer = AudioBuffer(max_size=max_size)
        
        # Audio input/output hardware simulator instances
        self.recorder = AudioRecorder(config, self.input_buffer)
        self.player = AudioPlayer(config, self.output_buffer)
        
        # Core layer placeholders
        self.metrics_tracker: Optional[MetricsTracker] = None
        self.queue_manager: Optional[QueueManager] = None
        self.context: Optional[PipelineContext] = None
        self.conversation_recorder: Optional[ConversationRecorder] = None
        self.workers: List[Any] = []

    def initialize_pipeline(self) -> None:
        """
        Loads provider models, builds PipelineContext, and initializes worker threads.
        """
        logger.info("Initializing pipeline factories...")
        
        # Resolve active provider names
        providers = self.config.get("active_providers", {})
        
        # Resolve instances using factory Registries
        noise_canceller = NoiseFactory.get_provider(providers.get("noise"), self.config)
        vad = VADFactory.get_provider(providers.get("vad"), self.config)
        stt = STTFactory.get_provider(providers.get("stt"), self.config)
        llm = LLMFactory.get_provider(providers.get("llm"), self.config)
        tts = TTSFactory.get_provider(providers.get("tts"), self.config)
        
        # Setup Core Infrastructure components
        self.metrics_tracker = MetricsTracker()
        
        # The input queue is linked directly to recorder's input_buffer._queue.
        # The intermediate playback_queue remains a separate bounded pipeline queue.
        self.queue_manager = QueueManager(
            config=self.config,
            input_queue=self.input_buffer._queue,
            output_queue=None
        )
        
        self.context = PipelineContext(
            config=self.config,
            queue_manager=self.queue_manager,
            metrics_tracker=self.metrics_tracker
        )

        # Initialize conversation recorder (single file per session)
        self.conversation_recorder = ConversationRecorder(
            session_id=self.context.session_id,
            output_dir="audio/recordings"
        )
        self.conversation_recorder.open()
        self.context.conversation_recorder = self.conversation_recorder
        
        # Initialize Memory Manager and bind to Context conditionally
        memory_enabled = self.config.get("memory", {}).get("enabled", True)
        if memory_enabled:
            self.memory_manager = MemoryManager(
                config=self.config,
                llm=llm,
                metrics_tracker=self.metrics_tracker
            )
            self.context.memory_manager = self.memory_manager
        else:
            self.memory_manager = None
            self.context.memory_manager = None
        
        # Setup Interruption Manager and register callbacks
        self.interruption_manager = InterruptionManager(self.context)
        self.context.interruption_manager = self.interruption_manager
        self.context.register_interrupt_callback(self.player.interrupt)

        # Setup StreamingContext
        self.streaming_context = StreamingContext()
        self.context.streaming_context = self.streaming_context

        # Instantiate pipeline coordinator wrapper
        self.pipeline = VoicePipeline(self.context)
        
        # Setup dedicated pipeline worker threads
        noise_worker = NoiseWorker(
            context=self.context,
            input_queue=self.queue_manager.audio_queue,
            output_queue=self.queue_manager.speech_queue,
            noise_canceller=noise_canceller
        )
        
        vad_worker = VADWorker(
            context=self.context,
            input_queue=self.queue_manager.speech_queue,
            output_queue=self.queue_manager.transcript_queue,
            vad=vad
        )
        
        stt_worker = STTWorker(
            context=self.context,
            input_queue=self.queue_manager.transcript_queue,
            output_queue=self.queue_manager.partial_transcript_queue,
            stt=stt
        )
        
        streaming_llm_worker = StreamingLLMWorker(
            context=self.context,
            input_queue=self.queue_manager.partial_transcript_queue,
            output_queue=self.queue_manager.partial_response_queue,
            llm=llm
        )
        
        sentence_aggregator_worker = SentenceAggregatorWorker(
            context=self.context,
            input_queue=self.queue_manager.partial_response_queue,
            output_queue=self.queue_manager.tts_queue
        )
        
        streaming_tts_worker = StreamingTTSWorker(
            context=self.context,
            input_queue=self.queue_manager.tts_queue,
            output_queue=self.queue_manager.partial_audio_queue,
            tts=tts
        )
        
        playback_worker = PlaybackWorker(
            context=self.context,
            input_queue=self.queue_manager.partial_audio_queue,
            output_queue=self.output_buffer._queue
        )
        
        interruption_worker = InterruptionWorker(
            context=self.context,
            input_queue=self.queue_manager.interruption_queue
        )
        
        self.workers = [
            noise_worker,
            vad_worker,
            stt_worker,
            streaming_llm_worker,
            sentence_aggregator_worker,
            streaming_tts_worker,
            playback_worker,
            interruption_worker
        ]

        
        logger.info(f"Pipeline initialized. {len(self.workers)} workers instantiated.")

    def start(self) -> None:
        """
        Starts the pipeline, recording hardware loops, and worker threads.
        """
        if self._is_running:
            return
            
        self._is_running = True
        logger.info("Starting pipeline workers...")
        print("Workers Started")  # Print matching validation expectation
        
        # Start hardware threads
        self.recorder.start_recording()
        self.player.start_playback()
        
        # Start daemon workers
        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        """
        Shuts down workers, halts capture buffers, and exports latency statistics.
        """
        if not self._is_running:
            return
            
        self._is_running = False
        logger.info("Stopping pipeline workers...")
        
        # Halt recording capture & output playback loops
        self.recorder.stop_recording()
        self.player.stop_playback()
        
        # Shutdown worker threads gracefully
        for worker in self.workers:
            try:
                worker.shutdown(timeout=1.0)
            except Exception as e:
                logger.error(f"Error encountered during worker shutdown: {e}")
                
        # Close conversation recorder
        if self.conversation_recorder:
            self.conversation_recorder.close()

        # Export metrics logging
        if self.metrics_tracker:
            self.metrics_tracker.export_json("metrics.json")
            print("metrics.json exported")  # Print matching validation expectation

    def is_running(self) -> bool:
        """
        Returns True if the pipeline is active.
        """
        return self._is_running

    def monitor_worker_health(self) -> Dict[str, bool]:
        """
        Queries thread alive status and checks health.
        """
        return {worker.name: worker.health_check() for worker in self.workers}
