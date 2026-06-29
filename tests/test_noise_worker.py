import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import AudioPayload
from core.pipeline_context import PipelineContext
from core.metrics import MetricsTracker
from core.queue_manager import QueueManager
from workers.noise_worker import NoiseWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"audio_queue_size": 5, "speech_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    return context, qm

def test_noise_worker_processing(mock_context_and_queues):
    """
    Verify that NoiseWorker pulls, processes, and pushes payloads correctly.
    """
    context, qm = mock_context_and_queues
    
    # Mock Noise Canceller
    mock_canceller = MagicMock()
    mock_canceller.process.return_value = b"\xaa\xbb"
    
    worker = NoiseWorker(
        context=context,
        input_queue=qm.audio_queue,
        output_queue=qm.speech_queue,
        noise_canceller=mock_canceller
    )
    
    # Create test payload
    payload = AudioPayload(request_id="req-1", audio=b"\x11\x22", created_at=time.time())
    qm.audio_queue.put(payload)
    
    # Run loop step manually
    worker.process_loop_step()
    
    # Check that output queue contains the processed payload
    assert qm.speech_queue.qsize() == 1
    out_payload = qm.speech_queue.get()
    assert isinstance(out_payload, AudioPayload)
    assert out_payload.audio == b"\xaa\xbb"
    assert out_payload.request_id == "req-1"
    mock_canceller.process.assert_called_once_with(b"\x11\x22")

def test_noise_worker_exception_isolation(mock_context_and_queues):
    """
    Verify that exceptions in processing are isolated and logged without crash.
    """
    context, qm = mock_context_and_queues
    
    mock_canceller = MagicMock()
    mock_canceller.process.side_effect = RuntimeError("Filtering crashed")
    
    worker = NoiseWorker(
        context=context,
        input_queue=qm.audio_queue,
        output_queue=qm.speech_queue,
        noise_canceller=mock_canceller
    )
    
    payload = AudioPayload(request_id="req-2", audio=b"\x11\x22", created_at=time.time())
    qm.audio_queue.put(payload)
    
    # Run thread worker manually
    worker.process_loop_step()
    
    # No output payload pushed, input task completed, error count incremented
    assert qm.speech_queue.empty() is True
    assert worker.error_count == 1
