import pytest
import time
from unittest.mock import MagicMock
from core.events import EventType
from pipeline.pipeline_state import PipelineState
from pipeline.pipeline_manager import PipelineManager
from workers.vad_worker import VADWorker

@pytest.fixture
def streaming_config():
    return {
        "active_providers": {
            "noise": "dummy",
            "vad": "dummy",
            "stt": "deepgram_streaming",
            "llm": "ollama_streaming",
            "tts": "kokoro_streaming"
        },
        "queues": {
            "audio_queue_size": 100,
            "speech_queue_size": 50,
            "transcript_queue_size": 10,
            "response_queue_size": 10,
            "tts_queue_size": 10,
            "playback_queue_size": 10
        },
        "audio": {
            "buffer_max_size": 100
        },
        "stt": {
            "deepgram": {"api_key": "MOCK_KEY"}
        },
        "llm": {
            "ollama": {"url": "http://localhost:11434/api/generate"}
        },
        "tts": {
            "kokoro": {"voice": "af_bella"}
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

@pytest.fixture(autouse=True)
def mock_network_calls(monkeypatch):
    import requests
    import urllib.request
    import urllib.error
    from unittest.mock import MagicMock
    import tts.providers.kokoro
    
    # Mock requests
    monkeypatch.setattr(
        requests,
        "post",
        MagicMock(side_effect=requests.exceptions.ConnectionError("Mocked connection failure"))
    )
    
    # Mock urllib
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        MagicMock(side_effect=urllib.error.URLError("Mocked connection failure"))
    )

    # Force Kokoro to fallback mode to avoid loading model weights and running slow ONNX inference
    monkeypatch.setattr(tts.providers.kokoro, "KOKORO_AVAILABLE", False)

def test_scenario_1_end_to_end_stream(streaming_config):
    """
    Scenario 1: Verify the complete streaming provider stack executes end-to-end.
    User Speaks -> Streaming STT -> Streaming LLM -> Sentence Aggregator -> Streaming TTS -> Playback.
    """
    manager = PipelineManager(streaming_config)
    manager.initialize_pipeline()
    
    # Disable player loop for direct validation
    manager.player.start_playback = MagicMock()
    
    manager.start()
    
    try:
        # Push mock frames
        manager.input_buffer.push(b"\x01\x02\x03\x04")
        time.sleep(0.01)
        
        # Push silent VAD frames to trigger completed speech segment
        for _ in range(5):
            manager.input_buffer.push(b"\x00" * 960)
            time.sleep(0.02)
            
        # Wait for audio chunks to reach output buffer
        timeout = 5.0
        start_time = time.time()
        completed = False
        while time.time() - start_time < timeout:
            if manager.output_buffer.size() > 0:
                completed = True
                break
            time.sleep(0.05)
            
        assert completed is True, "Scenario 1: Failed to propagate streaming data to output buffer."
        
        # Verify output PCM content
        chunk = manager.output_buffer.pop()
        assert chunk is not None
        assert len(chunk) > 0
        assert b"\x00" in chunk
        
        # Verify metric registration
        summary = manager.metrics_tracker.get_summary()
        assert summary["first_partial_transcript_ms"]["count"] >= 1.0
        assert summary["first_llm_token_ms"]["count"] >= 1.0
        assert summary["first_sentence_ms"]["count"] >= 1.0
        assert summary["first_audio_chunk_ms"]["count"] >= 1.0
        assert summary["total_turnaround_ms"]["count"] >= 1.0

    finally:
        manager.stop()

def test_scenario_2_interruption(streaming_config):
    """
    Scenario 2: Verify that during active speaking, a user interruption
    stops playback immediately, flushes streaming queues, and transitions state.
    """
    manager = PipelineManager(streaming_config)
    manager.initialize_pipeline()
    
    # Disable hardware loops
    manager.recorder.start_recording = MagicMock()
    manager.player.start_playback = MagicMock()
    
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
    
    manager.start()
    
    try:
        # Pre-fill streaming queues to verify flushing
        qm = manager.queue_manager
        qm.partial_transcript_queue.put("stt-item")
        qm.partial_response_queue.put("llm-item")
        qm.tts_queue.put("tts-item")
        qm.partial_audio_queue.put("audio-item")
        
        manager.output_buffer.push(b"\x00\x00")
        
        # Simulate active speaking state
        manager.context.set_state(PipelineState.SPEAKING)
        manager.context.set_active_request_id("req-old")
        
        # Reset VAD mock triggers
        vad_worker = next(w for w in manager.workers if isinstance(w, VADWorker))
        vad_worker.vad._triggered = False
        vad_worker._in_speech = False
        
        # Push mock audio frame to trigger VAD speech detected (barge-in)
        manager.input_buffer.push(b"\x01\x02\x03\x04")
        time.sleep(0.1)
        
        # Verify queues are empty
        assert qm.partial_transcript_queue.empty() is True
        assert qm.partial_response_queue.empty() is True
        assert qm.tts_queue.empty() is True
        assert qm.partial_audio_queue.empty() is True
        assert manager.output_buffer.size() == 0
        
        assert started is True
        assert finished is True
        assert manager.context.get_state() == PipelineState.LISTENING

    finally:
        manager.stop()
