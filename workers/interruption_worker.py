import logging
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import InterruptionPayload

logger = logging.getLogger(__name__)

class InterruptionWorker(BaseWorker):
    """
    Worker that retrieves InterruptionPayload from interruption_queue,
    and calls the InterruptionManager to handle the barge-in sequence.
    """

    def __init__(self, context: Any, input_queue: Any):
        """
        Initializes the InterruptionWorker.
        """
        super().__init__(name="InterruptionWorker", context=context, input_queue=input_queue)

    def process(self, payload: InterruptionPayload) -> None:
        """
        Process a single InterruptionPayload.
        """
        if not payload or not isinstance(payload, InterruptionPayload):
            logger.warning("Received invalid or empty payload in InterruptionWorker.")
            return

        logger.info(f"InterruptionWorker processing interruption for request {payload.request_id}")
        
        # Invoke InterruptionManager if present in context
        if hasattr(self.context, "interruption_manager") and self.context.interruption_manager:
            self.context.interruption_manager.handle_interruption(payload.request_id)
        else:
            logger.warning("InterruptionManager not found in context.")
