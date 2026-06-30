import pytest
import time
from unittest.mock import MagicMock
from core.events import EventType
from pipeline.pipeline_state import PipelineState
from pipeline.pipeline_manager import PipelineManager

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
            "audio_queue_size": 100,
            "speech_queue_size": 50,
            "transcript_queue_size": 10,
            "response_queue_size": 10,
            "tts_queue_size": 10,
            "playback_queue_size": 10
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

def test_streaming_pipeline_end_to_end_metrics(base_config):
    """
    Verify the complete Phase 4A streaming pipeline executes end-to-end,
    generating partial outputs, triggering playback, and recording all first-chunk latency metrics.
    """
    manager = PipelineManager(base_config)
    manager.initialize_pipeline()
    
    # Disable background player thread to inspect output buffer directly
    manager.player.start_playback = MagicMock()
    
    # Start workers
    manager.start()
    
    try:
        # Push initial frame to trigger speech start inside VADWorker
        manager.input_buffer.push(b"\x01\x02\x03\x04")
        time.sleep(0.01)
        
        # Verify state is active
        assert manager.context.get_state() in (
            PipelineState.LISTENING,
            PipelineState.PROCESSING,
            PipelineState.THINKING,
            PipelineState.SPEAKING,
            PipelineState.IDLE
        )
        
        # Push silent frames to trigger VAD silence boundary and final SpeechPayload
        for _ in range(5):
            manager.input_buffer.push(b"\x00" * 960)
            time.sleep(0.02)
            
        # Give worker threads a moment to execute
        # VAD -> STT -> LLM -> Sentence Aggregator -> TTS -> Playback -> output_buffer
        timeout = 5.0
        start_time = time.time()
        completed = False
        
        while time.time() - start_time < timeout:
            if manager.output_buffer.size() > 0:
                completed = True
                break
            time.sleep(0.05)
            
        assert completed is True, "Streaming pipeline failed to produce playback chunks in time."
        
        # Verify output audio
        chunk = manager.output_buffer.pop()
        assert chunk is not None
        assert len(chunk) > 0
        
        # Verify all Phase 4 telemetry metrics were recorded
        summary = manager.metrics_tracker.get_summary()
        assert summary["first_partial_transcript_ms"]["count"] >= 1.0
        assert summary["first_llm_token_ms"]["count"] >= 1.0
        assert summary["first_sentence_ms"]["count"] >= 1.0
        assert summary["first_audio_chunk_ms"]["count"] >= 1.0
        assert summary["ttft_ms"]["count"] >= 1.0
        assert summary["total_turnaround_ms"]["count"] >= 1.0

    finally:
        manager.stop()

def test_streaming_pipeline_interruption_flushing(base_config):
    """
    Verify that an interruption event resets context active request IDs
    and flushes all active streaming queues mid-turn.
    """
    manager = PipelineManager(base_config)
    manager.initialize_pipeline()
    
    # Disable background recorder and player loops to keep tests deterministic
    manager.recorder.start_recording = MagicMock()
    manager.player.start_playback = MagicMock()
    
    # Setup listeners
    started = False
    finished = False
    
    def on_started(req_id):
        nonlocal started
        started = True
        
    def on_finished(req_id):
        nonlocal finished
        finished = True
        
    manager.context.register_event_listener(EventType.INTERRUPTION_STARTED, on_started)
    manager.context.register_event_listener(EventType.INTERRUPTION_FINISHED, on_finished)

    # Start manager
    manager.start()
    
    try:
        # Pre-fill streaming queues
        qm = manager.queue_manager
        qm.partial_transcript_queue.put("item")
        qm.partial_response_queue.put("item")
        qm.tts_queue.put("item")
        qm.partial_audio_queue.put("item")
        
        manager.output_buffer.push(b"\x00\x00")
        
        # Set state to SPEAKING to simulate active agent synthesis
        manager.context.set_state(PipelineState.SPEAKING)
        manager.context.set_active_request_id("req-old")
        
        # Reset VAD trigger flag so it detects the barge-in frame as speech
        from workers.vad_worker import VADWorker
        vad_worker = next(w for w in manager.workers if isinstance(w, VADWorker))
        vad_worker.vad._triggered = False
        vad_worker._in_speech = False

        # User barge-in: trigger speech detection
        manager.input_buffer.push(b"\x01\x02\x03\x04")
        time.sleep(0.1) # Wait for VADWorker and InterruptionWorker to process

        # Verify queues are cleared and events are fired
        assert qm.partial_transcript_queue.empty() is True
        assert qm.partial_response_queue.empty() is True
        assert qm.tts_queue.empty() is True
        assert qm.partial_audio_queue.empty() is True
        assert manager.output_buffer.size() == 0  # Player buffer cleared
        
        assert started is True
        assert finished is True
        assert manager.context.get_state() == PipelineState.LISTENING

    finally:
        manager.stop()
