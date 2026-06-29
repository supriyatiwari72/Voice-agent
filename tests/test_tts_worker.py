import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import ResponsePayload, TTSPayload
from core.pipeline_context import PipelineContext
from core.metrics import MetricsTracker
from core.queue_manager import QueueManager
from pipeline.pipeline_state import PipelineState
from workers.tts_worker import TTSWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"tts_queue_size": 5, "playback_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    return context, qm

def test_tts_worker_synthesis(mock_context_and_queues):
    """
    Verify that TTSWorker generates speech audio, tracks TTS latency, and pushes TTSPayload.
    """
    context, qm = mock_context_and_queues
    
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = b"\x00\x01\x02"
    
    worker = TTSWorker(
        context=context,
        input_queue=qm.tts_queue,
        output_queue=qm.playback_queue,
        tts=mock_tts
    )
    
    payload = ResponsePayload(request_id="req-tts", response="Hello", user_done_timestamp=time.time())
    qm.tts_queue.put(payload)
    
    # Set active request ID so it doesn't get dropped as stale
    context.set_active_request_id("req-tts")
    
    worker.process_loop_step()
    
    # Assert state shift
    assert context.get_state() == PipelineState.SPEAKING
    
    # Assert output queue details
    assert qm.playback_queue.qsize() == 1
    out_payload = qm.playback_queue.get()
    assert isinstance(out_payload, TTSPayload)
    assert out_payload.audio == b"\x00\x01\x02"
    assert out_payload.request_id == "req-tts"
    
    # Assert metrics tracking
    summary = context.metrics.get_summary()
    assert summary["tts_latency_ms"]["count"] == 1.0
    assert summary["tts_latency_ms"]["average"] > 0.0
