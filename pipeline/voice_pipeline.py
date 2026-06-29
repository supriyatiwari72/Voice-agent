import logging
from typing import Dict, Any
from pipeline.pipeline_state import PipelineState
from core.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)

class VoicePipeline:
    """
    Coordinator class maintaining active config settings and state transitions.
    Heavy processing and AI model inference are delegated entirely to dedicated worker threads.
    """

    def __init__(self, context: PipelineContext):
        """
        Initializes the VoicePipeline coordinator with context.
        """
        self.context = context
        logger.info("VoicePipeline coordinator initialized.")

    def set_state(self, state: PipelineState) -> None:
        """
        Transitions the current pipeline state thread-safely.
        """
        self.context.set_state(state)

    def get_state(self) -> PipelineState:
        """
        Retrieves the active pipeline state.
        """
        return self.context.get_state()

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieves configuration settings.
        """
        return self.context.config
