import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

class BaseWorker(threading.Thread, ABC):
    """
    Abstract Base Worker establishing thread run loops, health checks,
    exception isolation, queue polling, and graceful shutdowns.
    """

    def __init__(
        self, 
        name: str, 
        context: Any, 
        input_queue: queue.Queue, 
        output_queue: Optional[queue.Queue] = None
    ):
        """
        Initializes the base worker thread.
        """
        # Enforce daemon thread support
        super().__init__(name=name, daemon=True)
        self.context = context
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.error_count = 0
        self.processed_count = 0
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """
        Starts the worker daemon thread.
        """
        self.start_time = time.time()
        logger.info(f"Starting worker thread: {self.name}")
        super().start()

    def stop(self) -> None:
        """
        Signals the run loop to terminate.
        """
        logger.info(f"Signaling worker to stop: {self.name}")
        self.stop_event.set()

    def run(self) -> None:
        """
        Executes the thread loop with exception isolation.
        """
        logger.info(f"Worker thread loop active: {self.name}")
        while not self.stop_event.is_set():
            try:
                self.process_loop_step()
            except Exception as e:
                logger.error(f"Fatal loop exception in worker '{self.name}': {e}", exc_info=True)
                time.sleep(0.1)
                
        logger.info(f"Worker thread loop terminated: {self.name}")

    def process_loop_step(self) -> None:
        """
        Processes a single step of the worker queue polling loop.
        """
        try:
            payload = self.input_queue.get(timeout=0.5)
            try:
                self.process(payload)
                self.processed_count += 1
            except Exception as e:
                self.error_count += 1
                logger.error(f"Isolated exception caught in worker '{self.name}': {e}", exc_info=True)
                self.handle_error(e)
                # Sleep briefly to avoid CPU starvation in case of rapid repeating errors
                time.sleep(0.1)
            finally:
                try:
                    self.input_queue.task_done()
                except ValueError:
                    # Benign race condition: queue was flushed externally (e.g. interruption
                    # manager reset unfinished_tasks=0) while this worker already held an
                    # item via get(). Safe to ignore — only occurs during shutdown/flush.
                    pass
        except queue.Empty:
            pass

    @abstractmethod
    def process(self, payload: Any) -> None:
        """
        Processes a single input queue payload chunk.
        Must be implemented by concrete subclasses.
        """
        pass

    def handle_error(self, e: Exception) -> None:
        """
        Custom error recovery hook. Can be overridden in concrete subclasses.
        """
        pass

    def health_check(self) -> bool:
        """
        Audits worker health status. Returns True if active and not crashed.
        """
        return self.is_alive() and not self.stop_event.is_set()

    def shutdown(self, timeout: float = 1.0) -> None:
        """
        Signals stopping and joins the thread with a timeout limit.
        """
        self.stop()
        self.join(timeout=timeout)
        if self.is_alive():
            logger.warning(f"Worker '{self.name}' failed to shut down within timeout of {timeout}s.")
        else:
            logger.info(f"Worker '{self.name}' shut down successfully.")
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns simple performance metrics for the worker thread.
        """
        uptime = time.time() - self.start_time if self.start_time else 0.0
        return {
            "name": self.name,
            "uptime": uptime,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "healthy": self.health_check()
        }
