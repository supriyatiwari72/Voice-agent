import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import SpeechPayload, PartialTranscriptPayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from core.streaming_context import StreamingContext
from workers.simulated_streaming_stt_worker import SimulatedStreamingSTTWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"transcript_queue_size": 5, "partial_transcript_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    context.streaming_context = StreamingContext()
    return context, qm

def test_simulated_streaming_stt_worker_flow(mock_context_and_queues):
    """
    Verify SimulatedStreamingSTTWorker splits full transcripts into word payloads.
    """
    context, qm = mock_context_and_queues
    
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "hello world"
    
    worker = SimulatedStreamingSTTWorker(
        context=context,
        input_queue=qm.transcript_queue,
        output_queue=qm.partial_transcript_queue,
        stt=mock_stt
    )
    
    user_done = time.time() - 0.1
    payload = SpeechPayload(request_id="req-stt", audio=b"\x00", user_done_timestamp=user_done)
    qm.transcript_queue.put(payload)
    
    context.set_active_request_id("req-stt")
    
    # Process
    worker.process_loop_step()
    
    # Expect 2 words in partial_transcript_queue
    assert qm.partial_transcript_queue.qsize() == 2
    
    p1 = qm.partial_transcript_queue.get()
    assert isinstance(p1, PartialTranscriptPayload)
    assert p1.text_chunk == "hello "
    assert p1.is_final is False
    assert p1.request_id == "req-stt"
    
    p2 = qm.partial_transcript_queue.get()
    assert isinstance(p2, PartialTranscriptPayload)
    assert p2.text_chunk == "world"
    assert p2.is_final is True
    assert p2.request_id == "req-stt"
    
    # Verify first partial transcript metric recorded
    summary = context.metrics.get_summary()
    assert summary["first_partial_transcript_ms"]["count"] == 1.0
    assert summary["first_partial_transcript_ms"]["average"] > 0.0
