import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import SentencePayload, PartialAudioPayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from core.streaming_context import StreamingContext
from workers.streaming_tts_worker import StreamingTTSWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"tts_queue_size": 5, "playback_queue_size": 5, "partial_audio_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    context.streaming_context = StreamingContext()
    return context, qm

def test_streaming_tts_worker_synthesis(mock_context_and_queues):
    """
    Verify StreamingTTSWorker synthesizes sentence chunks into audio payloads.
    """
    context, qm = mock_context_and_queues
    
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = b"\x10\x20"
    
    worker = StreamingTTSWorker(
        context=context,
        input_queue=qm.tts_queue,
        output_queue=qm.partial_audio_queue,
        tts=mock_tts
    )
    
    context.set_active_request_id("req-tts")
    
    payload = SentencePayload("req-tts", "Hello world.", is_final=True, user_done_timestamp=time.time())
    qm.tts_queue.put(payload)
    
    worker.process_loop_step()
    
    assert mock_tts.synthesize.call_count == 1
    assert qm.partial_audio_queue.qsize() == 1
    
    audio_out = qm.partial_audio_queue.get()
    assert isinstance(audio_out, PartialAudioPayload)
    assert audio_out.audio_chunk == b"\x10\x20"
    assert audio_out.is_final is True
    
    summary = context.metrics.get_summary()
    assert summary["tts_latency_ms"]["count"] == 1.0
