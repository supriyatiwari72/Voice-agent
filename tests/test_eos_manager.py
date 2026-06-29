import pytest
import time
from core.eos_manager import EOSManager
from pipeline.pipeline_state import PipelineState

@pytest.fixture
def base_config():
    return {
        "audio": {
            "frame_duration_ms": 30
        },
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "rms_threshold": 0.025
                }
            },
            "eos": {
                "speech_start_threshold": 0.60,
                "speech_end_threshold": 0.30,
                "consecutive_silence_frames": 5,  # short limit for testing: 150ms
                "min_speech_duration_ms": 100.0,  # 100ms limit
                "silence_timeout_ms": 200.0,      # 200ms limit
                "barge_in_rms_multiplier": 3.0
            }
        }
    }

def test_eos_manager_hysteresis_start(base_config):
    """Verify that speech only starts when confidence crosses speech_start_threshold."""
    manager = EOSManager(base_config)
    
    # 1. High RMS but low confidence (< 0.60)
    res = manager.process_frame(0.040, 0.50, PipelineState.LISTENING)
    assert res["speech_started"] is False
    assert res["is_speech_active"] is False

    # 2. Confidence crosses speech_start_threshold (0.60)
    res = manager.process_frame(0.040, 0.65, PipelineState.LISTENING)
    assert res["speech_started"] is True
    assert res["is_speech_active"] is True

def test_eos_manager_hysteresis_end(base_config):
    """Verify that speech is sustained when confidence is between thresholds."""
    manager = EOSManager(base_config)
    
    # Start speech
    manager.process_frame(0.040, 0.70, PipelineState.LISTENING)
    assert manager.in_speech is True
    
    # Confidence drops to 0.40 (above end threshold 0.30)
    res = manager.process_frame(0.040, 0.40, PipelineState.LISTENING)
    assert res["is_speech_active"] is True
    assert res["speech_ended"] is False

    # Confidence drops below 0.30
    res = manager.process_frame(0.040, 0.25, PipelineState.LISTENING)
    assert res["is_speech_active"] is True  # still active, waiting for silence timeout
    assert res["speech_ended"] is False

def test_eos_manager_consecutive_silence_timeout(base_config):
    """Verify speech ends after consecutive silent frames limit is met."""
    manager = EOSManager(base_config)
    
    # Start speech
    manager.process_frame(0.040, 0.70, PipelineState.LISTENING)
    time.sleep(0.15) # Wait to exceed min speech duration (100ms)
    
    # 4 consecutive silent frames (less than 5)
    for _ in range(4):
        res = manager.process_frame(0.005, 0.10, PipelineState.LISTENING)
        assert res["is_speech_active"] is True
        assert res["speech_ended"] is False
        
    # 5th silent frame triggers EOS endpoint
    res = manager.process_frame(0.005, 0.10, PipelineState.LISTENING)
    assert res["is_speech_active"] is False
    assert res["speech_ended"] is True

def test_eos_manager_min_speech_duration_discard(base_config):
    """Verify short speech bursts are rejected."""
    manager = EOSManager(base_config)
    
    # Start speech
    manager.process_frame(0.040, 0.70, PipelineState.LISTENING)
    
    # End speech immediately (less than 100ms duration)
    for _ in range(5):
        res = manager.process_frame(0.005, 0.10, PipelineState.LISTENING)
        
    # Endpoint met, but discarded since duration < 100ms
    assert manager.in_speech is False
    assert res["is_speech_active"] is False
    assert res["speech_ended"] is False  # Discarded (no speech_ended trigger)

def test_eos_manager_adaptive_noise_floor(base_config):
    """Verify noise floor adjusts based on low confidence frames."""
    manager = EOSManager(base_config)
    
    initial_floor = manager.noise_floor
    assert initial_floor == 0.025
    
    # Feed multiple quiet frames with low confidence
    for _ in range(20):
        manager.process_frame(0.015, 0.05, PipelineState.LISTENING)
        
    # Noise floor should adapt downward toward 0.015
    assert manager.noise_floor < initial_floor
    assert 0.012 <= manager.noise_floor <= 0.018
