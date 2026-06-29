import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import TranscriptPayload, ResponsePayload
from core.pipeline_context import PipelineContext
from core.metrics import MetricsTracker
from core.queue_manager import QueueManager
from pipeline.pipeline_state import PipelineState
from workers.llm_worker import LLMWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"response_queue_size": 5, "tts_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    return context, qm

def test_llm_worker_generation(mock_context_and_queues):
    """
    Verify that LLMWorker processes input stream tokens, tracks TTFT/overall latency,
    and forwards the consolidated ResponsePayload.
    """
    context, qm = mock_context_and_queues
    
    mock_llm = MagicMock()
    # Mock generation stream: yields three tokens with minor sleeps to simulate delay
    def mock_stream(text):
        time.sleep(0.01)
        yield "Hello"
        time.sleep(0.01)
        yield ", how can I help"
        time.sleep(0.01)
        yield " you?"
        
    mock_llm.generate_stream.side_effect = mock_stream
    
    worker = LLMWorker(
        context=context,
        input_queue=qm.response_queue,
        output_queue=qm.tts_queue,
        llm=mock_llm
    )
    
    payload = TranscriptPayload(request_id="req-llm", text="hello", user_done_timestamp=time.time())
    qm.response_queue.put(payload)
    
    # Set active request ID so it doesn't get dropped as stale
    context.set_active_request_id("req-llm")
    
    worker.process_loop_step()
    
    # Assert state shift
    assert context.get_state() == PipelineState.THINKING
    
    # Assert output queue details
    assert qm.tts_queue.qsize() == 1
    out_payload = qm.tts_queue.get()
    assert isinstance(out_payload, ResponsePayload)
    assert out_payload.response == "Hello, how can I help you?"
    assert out_payload.request_id == "req-llm"
    
    # Assert metrics tracking
    summary = context.metrics.get_summary()
    assert summary["ttft_ms"]["count"] == 1.0
    assert summary["ttft_ms"]["average"] > 0.0
    assert summary["llm_latency_ms"]["count"] == 1.0
    assert summary["llm_latency_ms"]["average"] > summary["ttft_ms"]["average"]
