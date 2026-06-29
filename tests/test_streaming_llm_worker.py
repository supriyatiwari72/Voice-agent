import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import PartialTranscriptPayload, PartialResponsePayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from core.streaming_context import StreamingContext
from workers.streaming_llm_worker import StreamingLLMWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"partial_transcript_queue_size": 5, "partial_response_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    context.streaming_context = StreamingContext()
    return context, qm

def test_streaming_llm_worker_generation(mock_context_and_queues):
    """
    Verify StreamingLLMWorker aggregates incoming chunks and triggers token streams on is_final.
    """
    context, qm = mock_context_and_queues
    
    mock_llm = MagicMock()
    def mock_stream(text):
        yield "token1"
        yield "token2"
    mock_llm.generate_stream.side_effect = mock_stream
    
    worker = StreamingLLMWorker(
        context=context,
        input_queue=qm.partial_transcript_queue,
        output_queue=qm.partial_response_queue,
        llm=mock_llm
    )
    
    context.set_active_request_id("req-llm")
    
    # 1. Push non-final chunk
    p1 = PartialTranscriptPayload("req-llm", "hello ", is_final=False, timestamp=time.time())
    qm.partial_transcript_queue.put(p1)
    
    worker.process_loop_step()
    # Output should still be empty (not final yet)
    assert qm.partial_response_queue.empty() is True
    assert worker._accumulated_transcripts["req-llm"] == "hello "
    
    # 2. Push final chunk
    p2 = PartialTranscriptPayload("req-llm", "world", is_final=True, timestamp=time.time())
    qm.partial_transcript_queue.put(p2)
    
    worker.process_loop_step()
    
    # Verify stream is processed
    assert mock_llm.generate_stream.call_count == 1
    mock_llm.generate_stream.assert_called_with("hello world")
    
    # Expect: token1, token2, and closing payload (total 3 payloads)
    assert qm.partial_response_queue.qsize() == 3
    
    res1 = qm.partial_response_queue.get()
    assert res1.token_chunk == "token1"
    assert res1.is_final is False
    
    res2 = qm.partial_response_queue.get()
    assert res2.token_chunk == "token2"
    assert res2.is_final is False
    
    res3 = qm.partial_response_queue.get()
    assert res3.token_chunk == ""
    assert res3.is_final is True
    
    # Verify metrics
    summary = context.metrics.get_summary()
    assert summary["ttft_ms"]["count"] == 1.0
    assert summary["first_llm_token_ms"]["count"] == 1.0
