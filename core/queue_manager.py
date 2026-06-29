import queue
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DiagnosticQueue(queue.Queue):
    """
    Queue wrapper collecting put/get wait times, depths, occupancy,
    starvation events, and overflow warnings.
    """

    def __init__(self, maxsize: int = 0, name: str = "diagnostic_queue"):
        super().__init__(maxsize=maxsize)
        self.name = name
        self.put_wait_times = []
        self.get_wait_times = []
        self.starvation_events = 0
        self.overflow_warnings = 0

    def put(self, item, block=True, timeout=None):
        size_before = self.qsize()
        max_size = self.maxsize
        if max_size > 0 and size_before >= int(max_size * 0.8):
            self.overflow_warnings += 1
            logger.warning(
                f"[Queue Diagnostic Warning] Queue '{self.name}' is near occupancy limit: "
                f"{size_before}/{max_size} ({(size_before/max_size)*100:.1f}%)"
            )
        
        start_time = time.time()
        super().put(item, block=block, timeout=timeout)
        wait_time = time.time() - start_time
        self.put_wait_times.append(wait_time)

    def get(self, block=True, timeout=None):
        if self.empty() and block:
            self.starvation_events += 1
            logger.debug(f"[Queue Diagnostic Info] Starvation event on empty queue '{self.name}'. Waiting for producer...")
        
        start_time = time.time()
        try:
            item = super().get(block=block, timeout=timeout)
            wait_time = time.time() - start_time
            self.get_wait_times.append(wait_time)
            return item
        except queue.Empty:
            wait_time = time.time() - start_time
            self.get_wait_times.append(wait_time)
            raise


class QueueManager:
    """
    Manages and monitors thread-safe pipeline queue communication channels.
    """

    def __init__(
        self, 
        config: Dict[str, Any], 
        input_queue: Optional[queue.Queue] = None, 
        output_queue: Optional[queue.Queue] = None
    ):
        """
        Initializes QueueManager, wiring hardware buffers if provided.
        """
        self.config = config or {}
        q_config = self.config.get("queues", {})

        # Load queue sizes from config or default to standard bounded limits
        audio_size = q_config.get("audio_queue_size", 200)
        speech_size = q_config.get("speech_queue_size", 100)
        transcript_size = q_config.get("transcript_queue_size", 50)
        response_size = q_config.get("response_queue_size", 50)
        tts_size = q_config.get("tts_queue_size", 50)
        playback_size = q_config.get("playback_queue_size", 50)

        self.audio_queue = input_queue if input_queue is not None else queue.Queue(maxsize=audio_size)
        self.speech_queue = queue.Queue(maxsize=speech_size)
        self.transcript_queue = DiagnosticQueue(maxsize=transcript_size, name="transcript_queue")
        self.response_queue = queue.Queue(maxsize=response_size)
        self.tts_queue = queue.Queue(maxsize=tts_size)
        self.playback_queue = output_queue if output_queue is not None else queue.Queue(maxsize=playback_size)
        self.interruption_queue = queue.Queue(maxsize=10)
        self.partial_transcript_queue = DiagnosticQueue(maxsize=transcript_size, name="partial_transcript_queue")
        self.partial_response_queue = DiagnosticQueue(maxsize=response_size, name="partial_response_queue")
        self.partial_audio_queue = queue.Queue(maxsize=playback_size)

        self._all_queues = {
            "audio_queue": self.audio_queue,
            "speech_queue": self.speech_queue,
            "transcript_queue": self.transcript_queue,
            "response_queue": self.response_queue,
            "tts_queue": self.tts_queue,
            "playback_queue": self.playback_queue,
            "interruption_queue": self.interruption_queue,
            "partial_transcript_queue": self.partial_transcript_queue,
            "partial_response_queue": self.partial_response_queue,
            "partial_audio_queue": self.partial_audio_queue
        }

        logger.info("QueueManager initialized with bounded queues.")

    def flush_all(self) -> None:
        """
        Flushes all items from all queues.
        """
        logger.info("Flushing all pipeline communication queues.")
        for name in self._all_queues.keys():
            self.flush_queue(name)

    def flush_queue(self, queue_name: str) -> None:
        """
        Flushes all items from a single queue in a thread-safe manner.
        """
        q = self._all_queues.get(queue_name)
        if q:
            with q.mutex:
                q.queue.clear()
                q.all_tasks_done.notify_all()
                q.unfinished_tasks = 0
            logger.debug(f"Flushed queue: {queue_name}")

    def reset_all(self) -> None:
        """
        Resets and clears all queues to baseline states.
        """
        logger.info("Resetting all pipeline queues.")
        self.flush_all()

    def get_occupancy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Computes occupancy statistics for health diagnostics.
        """
        metrics = {}
        for name, q in self._all_queues.items():
            size = q.qsize()
            maxsize = q.maxsize
            occupancy = (size / maxsize) if maxsize > 0 else 0.0
            metrics[name] = {
                "size": size,
                "maxsize": maxsize,
                "occupancy": occupancy
            }
        return metrics

    def monitor_health(self) -> Dict[str, str]:
        """
        Audits queue fill levels to detect stuck consumers or bottleneck overflows.
        Triggers warning logs when occupancy exceeds 80%.
        """
        health = {}
        for name, q in self._all_queues.items():
            size = q.qsize()
            maxsize = q.maxsize
            
            if q.full():
                health[name] = "FULL (OVERFLOW RISK)"
                logger.error(f"Queue OVERFLOW warning: '{name}' is FULL.")
            elif maxsize > 0 and size > int(maxsize * 0.8):
                health[name] = "WARNING (OCCUPANCY >80%)"
                logger.warning(f"Queue bottleneck warning: '{name}' usage exceeds 80% ({size}/{maxsize}).")
            else:
                health[name] = "HEALTHY"
        return health
