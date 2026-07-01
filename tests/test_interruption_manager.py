import pytest
import queue
import time
from unittest.mock import MagicMock
from core.events import EventType
from core.payloads import AudioPayload, InterruptionPayload, TranscriptPayload, ResponsePayload, TTSPayload, PartialResponsePayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from core.interruption_manager import InterruptionManager
from pipeline.pipeline_state import PipelineState
from workers.interruption_worker import InterruptionWorker
from workers.llm_worker import LLMWorker
from workers.tts_worker import TTSWorker
from workers.playback_worker import PlaybackWorker

@pytest.fixture
def mock_context_and_queues():
    config = {
        "queues": {
            "audio_queue_size": 10,
            "speech_queue_size": 10,
            "transcript_queue_size": 10,
            "response_queue_size": 10,
            "tts_queue_size": 10,
            "playback_queue_size": 10
        }
    }
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    return context, qm

def test_interruption_manager_flushing_and_state(mock_context_and_queues):
    """
    Verify InterruptionManager flushes intermediate queues, resets state, and calls interruption callbacks.
    """
    context, qm = mock_context_and_queues
    im = InterruptionManager(context)
    context.interruption_manager = im

    # Register mock interrupt callback
    mock_cb = MagicMock()
    context.register_interrupt_callback(mock_cb)

    # Listeners for events
    started_triggered = False
    finished_triggered = False

    def on_started(payload):
        nonlocal started_triggered
        started_triggered = True
        assert context.get_state() == PipelineState.INTERRUPTED

    def on_finished(payload):
        nonlocal finished_triggered
        finished_triggered = True
        assert context.get_state() == PipelineState.LISTENING

    context.register_event_listener(EventType.INTERRUPTION_STARTED, on_started)
    context.register_event_listener(EventType.INTERRUPTION_FINISHED, on_finished)

    # Pre-fill queues to test flushing
    qm.transcript_queue.put(TranscriptPayload("req-1", "hello", time.time()))
    qm.partial_response_queue.put(PartialResponsePayload("req-1", "hi", is_final=False, timestamp=time.time()))
    qm.tts_queue.put(TTSPayload("req-1", b"\x00", time.time()))
    qm.playback_queue.put(TTSPayload("req-1", b"\x00", time.time()))

    assert qm.transcript_queue.qsize() == 1
    assert qm.playback_queue.qsize() == 1

    # Trigger interruption
    context.set_state(PipelineState.SPEAKING)
    im.handle_interruption("req-1")

    # Verify downstream queues are empty, but STT queues are preserved
    assert qm.transcript_queue.qsize() == 1
    assert qm.partial_response_queue.empty() is True
    assert qm.tts_queue.empty() is True
    assert qm.playback_queue.empty() is True

    # Verify state transitions and callbacks
    assert mock_cb.call_count == 1
    assert started_triggered is True
    assert finished_triggered is True
    assert context.get_state() == PipelineState.LISTENING

def test_workers_abort_on_interruption_event(mock_context_and_queues):
    """
    Verify LLMWorker, TTSWorker, and PlaybackWorker drop/abort stale requests when interruption event is set.
    """
    context, qm = mock_context_and_queues
    context.set_active_request_id("req-new")
    context.interruption_event.set()

    # 1. Test LLMWorker abort
    mock_llm = MagicMock()
    llm_worker = LLMWorker(context, qm.response_queue, qm.tts_queue, mock_llm)
    
    # Process a stale request payload
    stale_payload = TranscriptPayload("req-old", "hello", time.time())
    qm.response_queue.put(stale_payload)
    
    llm_worker.process_loop_step()
    
    # Should not call llm.generate_stream or add to tts_queue
    assert mock_llm.generate_stream.call_count == 0
    assert qm.tts_queue.empty() is True

    # 2. Test TTSWorker abort
    mock_tts = MagicMock()
    tts_worker = TTSWorker(context, qm.tts_queue, qm.playback_queue, mock_tts)
    
    stale_resp = ResponsePayload("req-old", "hi", time.time())
    qm.tts_queue.put(stale_resp)
    
    tts_worker.process_loop_step()
    
    assert mock_tts.synthesize.call_count == 0
    assert qm.playback_queue.empty() is True

    # 3. Test PlaybackWorker abort
    playback_out = queue.Queue()
    playback_worker = PlaybackWorker(context, qm.playback_queue, playback_out)
    
    stale_tts = TTSPayload("req-old", b"\x00", time.time())
    qm.playback_queue.put(stale_tts)
    
    playback_worker.process_loop_step()
    
    assert playback_out.empty() is True

def test_interruption_manager_manual_interruption(mock_context_and_queues):
    """
    Verify InterruptionManager sets state to IDLE when is_manual=True.
    """
    context, qm = mock_context_and_queues
    im = InterruptionManager(context)
    context.interruption_manager = im

    context.set_state(PipelineState.SPEAKING)
    im.handle_interruption("req-1", is_manual=True)

    assert context.get_state() == PipelineState.IDLE
