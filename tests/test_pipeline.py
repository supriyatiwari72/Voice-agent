from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from pipeline.voice_pipeline import VoicePipeline
from pipeline.pipeline_state import PipelineState
from pipeline.pipeline_manager import PipelineManager

def test_pipeline_initialization_and_states():
    """
    Test that VoicePipeline correctly constructs with a PipelineContext and switches state.
    """
    config = {
        "queues": {
            "audio_queue_size": 10,
            "speech_queue_size": 10,
            "transcript_queue_size": 5,
            "response_queue_size": 5,
            "tts_queue_size": 5,
            "playback_queue_size": 10
        }
    }
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)

    # 1. Build Pipeline coordinator
    pipeline = VoicePipeline(context)

    # 2. Verify setup state
    assert pipeline.get_state() == PipelineState.INITIALIZING
    context.set_state(PipelineState.IDLE)
    assert pipeline.get_state() == PipelineState.IDLE

    # 3. Toggle states and verify behavior
    pipeline.set_state(PipelineState.LISTENING)
    assert pipeline.get_state() == PipelineState.LISTENING

    pipeline.set_state(PipelineState.SPEAKING)
    assert pipeline.get_state() == PipelineState.SPEAKING

def test_pipeline_manager_orchestration():
    """
    Test that PipelineManager initializes context, queues, and workers cleanly.
    """
    config = {
        "active_providers": {
            "noise": "dummy",
            "vad": "dummy",
            "stt": "dummy",
            "llm": "dummy",
            "tts": "dummy"
        },
        "queues": {
            "audio_queue_size": 10,
            "speech_queue_size": 10,
            "transcript_queue_size": 5,
            "response_queue_size": 5,
            "tts_queue_size": 5,
            "playback_queue_size": 10
        }
    }

    manager = PipelineManager(config)
    manager.initialize_pipeline()

    assert manager.context is not None
    assert manager.queue_manager is not None
    assert manager.metrics_tracker is not None
    assert len(manager.workers) == 8

    # Verify coordinator state transitions through context are correct
    assert manager.pipeline.get_state() == PipelineState.READY
