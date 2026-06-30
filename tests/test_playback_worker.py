import pytest
import queue
import time
from core.payloads import TTSPayload, PartialAudioPayload
from core.pipeline_context import PipelineContext
from core.metrics import MetricsTracker
from core.queue_manager import QueueManager
from pipeline.pipeline_state import PipelineState
from workers.playback_worker import PlaybackWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"playback_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    return context, qm

def test_playback_worker_playback(mock_context_and_queues):
    """
    Verify that PlaybackWorker pushes audio data, records turnaround metric,
    and transitions state to IDLE.
    """
    context, qm = mock_context_and_queues
    
    # Mock player output queue (acting as output_queue)
    player_queue = queue.Queue(maxsize=5)
    
    worker = PlaybackWorker(
        context=context,
        input_queue=qm.partial_audio_queue,
        output_queue=player_queue
    )
    
    user_done = time.time() - 0.5  # 500 ms ago
    payload = PartialAudioPayload(request_id="req-play", audio_chunk=b"\x00\x01", is_final=True, timestamp=user_done)
    qm.partial_audio_queue.put(payload)
    
    # Set active request ID so it doesn't get dropped as stale
    context.set_active_request_id("req-play")
    
    # Set state to SPEAKING to verify transition back to IDLE
    context.set_state(PipelineState.SPEAKING)
    
    # Setup streaming context tracking on context
    from core.streaming_context import StreamingContext
    context.streaming_context = StreamingContext()

    worker.process_loop_step()
    
    # Assert state shift back to IDLE
    assert context.get_state() == PipelineState.IDLE
    
    # Assert output player queue details
    assert player_queue.qsize() == 1
    out_audio = player_queue.get()
    assert out_audio == b"\x00\x01"
    
    # Assert metrics tracking
    summary = context.metrics.get_summary()
    assert summary["total_turnaround_ms"]["count"] == 1.0
    # Turnaround latency must be around 500ms (0.5s)
    assert summary["total_turnaround_ms"]["average"] >= 450.0

def test_playback_interruption_during_partial_audio(mock_context_and_queues):
    """
    Verify that if an interruption occurs during streaming playback,
    subsequent audio chunks for the interrupted request are dropped and never played.
    """
    context, qm = mock_context_and_queues
    player_queue = queue.Queue(maxsize=5)
    
    worker = PlaybackWorker(
        context=context,
        input_queue=qm.partial_audio_queue,
        output_queue=player_queue
    )
    
    context.set_active_request_id("req-play-interrupt")
    context.set_state(PipelineState.SPEAKING)
    
    from core.streaming_context import StreamingContext
    context.streaming_context = StreamingContext()
    
    # 1. Play first chunk
    chunk1 = PartialAudioPayload(request_id="req-play-interrupt", audio_chunk=b"\x01", is_final=False, timestamp=time.time())
    qm.partial_audio_queue.put(chunk1)
    worker.process_loop_step()
    
    assert player_queue.qsize() == 1
    assert player_queue.get() == b"\x01"
    
    # 2. Trigger interruption and request cancellation
    context.cancel_request("req-play-interrupt")
    context.interruption_event.set()
    
    # 3. Try to play subsequent chunk (it should be dropped)
    chunk2 = PartialAudioPayload(request_id="req-play-interrupt", audio_chunk=b"\x02", is_final=False, timestamp=time.time())
    qm.partial_audio_queue.put(chunk2)
    worker.process_loop_step()
    
    assert player_queue.empty() is True
    
    # 4. Try to play final chunk (it should also be dropped)
    chunk3 = PartialAudioPayload(request_id="req-play-interrupt", audio_chunk=b"\x03", is_final=True, timestamp=time.time())
    qm.partial_audio_queue.put(chunk3)
    worker.process_loop_step()
    
    assert player_queue.empty() is True
