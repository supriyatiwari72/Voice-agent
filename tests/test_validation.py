import pytest
import time
import queue
import logging
import threading
from unittest.mock import MagicMock, patch
from core.payloads import AudioPayload, SpeechPayload, TranscriptPayload, ResponsePayload, TTSPayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from pipeline.pipeline_manager import PipelineManager
from workers.noise_worker import NoiseWorker
from workers.vad_worker import VADWorker
from workers.stt_worker import STTWorker
from workers.llm_worker import LLMWorker
from workers.tts_worker import TTSWorker
from workers.playback_worker import PlaybackWorker

logger = logging.getLogger(__name__)

@pytest.fixture
def base_config():
    return {
        "active_providers": {
            "noise": "dummy",
            "vad": "dummy",
            "stt": "dummy",
            "llm": "dummy",
            "tts": "dummy"
        },
        "queues": {
            "audio_queue_size": 200,
            "speech_queue_size": 100,
            "transcript_queue_size": 50,
            "response_queue_size": 50,
            "tts_queue_size": 50,
            "playback_queue_size": 50
        },
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 1,
                    "min_speech_bytes": 0
                }
            }
        }
    }

def test_queue_stress_testing():
    """
    Stress test QueueManager by injecting 500, 1000, and 5000 items.
    Verify health warnings trigger correctly when occupancy > 80%.
    """
    config = {
        "queues": {
            "audio_queue_size": 10000,
            "speech_queue_size": 10,
            "transcript_queue_size": 10,
            "response_queue_size": 10,
            "tts_queue_size": 10,
            "playback_queue_size": 10
        }
    }
    qm = QueueManager(config)

    # 1. Verify health warnings at >80% usage
    # speech_queue has maxsize 10. 9 items is 90% (>80%)
    for i in range(9):
        qm.speech_queue.put(f"item-{i}")
    health = qm.monitor_health()
    assert health["speech_queue"] == "WARNING (OCCUPANCY >80%)"

    # Fill completely
    qm.speech_queue.put("item-9")
    health = qm.monitor_health()
    assert health["speech_queue"] == "FULL (OVERFLOW RISK)"

    # 2. Inject 500, 1000, 5000 items into audio_queue
    for count in [500, 1000, 5000]:
        t0 = time.time()
        for i in range(count):
            payload = AudioPayload(request_id=f"req-{i}", audio=b"\x00"*960, created_at=time.time())
            qm.audio_queue.put(payload)
        
        # Verify sizes
        assert qm.audio_queue.qsize() == count
        
        # Flush to prevent actual memory build-up in test
        qm.flush_queue("audio_queue")
        assert qm.audio_queue.qsize() == 0
        logger.info(f"Successfully injected and flushed {count} items in {time.time() - t0:.4f}s")

def test_worker_failure_isolation(base_config):
    """
    Verify that workers isolate exceptions thrown inside process(),
    incrementing their error count while remaining alive and active.
    """
    metrics = MetricsTracker()
    qm = QueueManager(base_config)
    context = PipelineContext(base_config, qm, metrics)

    # Instantiate VAD worker
    mock_vad = MagicMock()
    worker = VADWorker(context, qm.speech_queue, qm.transcript_queue, mock_vad)

    # Inject exception raising mock process method
    worker.process = MagicMock(side_effect=RuntimeError("Simulated Exception"))

    # Put item in queue and step loop
    payload = AudioPayload("req-fail", b"\x00"*960, time.time())
    qm.speech_queue.put(payload)
    
    worker.process_loop_step()

    # Worker should log the exception, increment errors, but remain active
    assert worker.error_count == 1
    assert worker.stop_event.is_set() is False
    assert qm.speech_queue.empty() is True

def test_shutdown_safety(base_config):
    """
    Verify that starting and immediately stopping the pipeline halts
    all worker threads and flushes all active queues cleanly.
    """
    manager = PipelineManager(base_config)
    manager.initialize_pipeline()
    
    # Start all workers
    manager.start()
    assert manager.is_running() is True
    
    # Confirm workers are alive
    for w in manager.workers:
        assert w.is_alive() is True

    # Immediately stop
    manager.stop()
    assert manager.is_running() is False

    # Verify all threads terminate
    for w in manager.workers:
        w.join(timeout=1.0)
        assert w.is_alive() is False

def test_long_duration_stability_simulation(base_config):
    """
    Simulate a conversation session over time.
    Monitor queues, thread count stability, and metrics logging.
    """
    manager = PipelineManager(base_config)
    manager.initialize_pipeline()
    manager.start()

    try:
        initial_threads = threading.active_count()
        
        # Simulate active speaker turns
        for turn in range(5):
            # User speaks (VAD gets speech chunk)
            manager.input_buffer.push(b"\x01\x02\x03\x04")
            time.sleep(0.02)
            # User stops speaking (VAD gets silent chunks)
            for _ in range(5):
                manager.input_buffer.push(b"\x00" * 960)
                time.sleep(0.02)

        # Let the pipeline process turns
        time.sleep(0.5)

        # Audit thread leaks
        current_threads = threading.active_count()
        # Verify thread count hasn't leaked exponentially
        assert current_threads <= initial_threads + 5

        # Audit queues to check for stagnation/build-up
        qm_metrics = manager.queue_manager.get_occupancy_metrics()
        for q_name, metrics in qm_metrics.items():
            # Standard worker processing should leave queues clean/empty after turns
            assert metrics["size"] <= 5, f"Queue {q_name} has build-up size: {metrics['size']}"

    finally:
        manager.stop()
