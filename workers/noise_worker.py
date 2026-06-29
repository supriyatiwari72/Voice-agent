import logging
import queue
import time
import uuid
from typing import Any
from core.worker_base import BaseWorker
from core.payloads import AudioPayload

logger = logging.getLogger(__name__)

class NoiseWorker(BaseWorker):
    """
    Worker that retrieves raw audio bytes, wraps them in AudioPayload,
    runs noise cancellation, and forwards the cleaned AudioPayload to the speech_queue.
    """

    def __init__(self, context: Any, input_queue: Any, output_queue: Any, noise_canceller: Any):
        """
        Initializes the NoiseWorker.
        """
        super().__init__(name="NoiseWorker", context=context, input_queue=input_queue, output_queue=output_queue)
        self.noise_canceller = noise_canceller

    def process_loop_step(self) -> None:
        """
        Processes a single step of the worker queue polling loop.
        Accepts raw bytes from the recorder queue and maps them to an AudioPayload.
        """
        try:
            popped = self.input_queue.get(timeout=0.5)
            try:
                if isinstance(popped, AudioPayload):
                    payload = popped
                elif isinstance(popped, bytes):
                    # Wrap raw audio bytes in AudioPayload with a unique request tracker ID
                    payload = AudioPayload(
                        request_id=f"req-{uuid.uuid4()}",
                        audio=popped,
                        created_at=time.time()
                    )
                else:
                    logger.warning(f"Unexpected item type in audio_queue: {type(popped)}")
                    return

                self.process(payload)
                self.processed_count += 1
            except Exception as e:
                self.error_count += 1
                logger.error(f"Isolated exception caught in worker '{self.name}': {e}", exc_info=True)
                self.handle_error(e)
                time.sleep(0.1)
            finally:
                self.input_queue.task_done()
        except queue.Empty:
            pass

    def process(self, payload: AudioPayload) -> None:
        """
        Process a single AudioPayload.
        """
        if not payload or not isinstance(payload, AudioPayload):
            logger.warning("Received invalid or empty payload in NoiseWorker.")
            return

        cleaned_audio = self.noise_canceller.process(payload.audio)
        
        # Build cleaned AudioPayload keeping the request_id and created_at timestamps
        cleaned_payload = AudioPayload(
            request_id=payload.request_id,
            audio=cleaned_audio,
            created_at=payload.created_at
        )

        if self.output_queue:
            self.output_queue.put(cleaned_payload)
