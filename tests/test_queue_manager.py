import pytest
import queue
from core.queue_manager import QueueManager

@pytest.fixture
def mock_config():
    return {
        "queues": {
            "audio_queue_size": 10,
            "speech_queue_size": 10,
            "transcript_queue_size": 5,
            "response_queue_size": 5,
            "tts_queue_size": 5,
            "playback_queue_size": 10
        }
    }

def test_queue_manager_creation(mock_config):
    """
    Verify that all bounded queues are initialized correctly.
    """
    qm = QueueManager(mock_config)
    assert qm.audio_queue.maxsize == 10
    assert qm.speech_queue.maxsize == 10
    assert qm.transcript_queue.maxsize == 5
    assert qm.response_queue.maxsize == 5
    assert qm.tts_queue.maxsize == 5
    assert qm.playback_queue.maxsize == 10

def test_queue_manager_flush(mock_config):
    """
    Verify that flush_queue clears elements from a specific queue.
    """
    qm = QueueManager(mock_config)
    
    # Push dummy elements
    qm.speech_queue.put("item1")
    qm.speech_queue.put("item2")
    assert qm.speech_queue.qsize() == 2
    
    qm.flush_queue("speech_queue")
    assert qm.speech_queue.qsize() == 0
    assert qm.speech_queue.empty() is True

def test_queue_manager_flush_all(mock_config):
    """
    Verify that flush_all clears elements from all queues.
    """
    qm = QueueManager(mock_config)
    
    qm.audio_queue.put(b"\x00")
    qm.response_queue.put("test prompt")
    
    assert qm.audio_queue.qsize() == 1
    assert qm.response_queue.qsize() == 1
    
    qm.flush_all()
    assert qm.audio_queue.qsize() == 0
    assert qm.response_queue.qsize() == 0

def test_queue_manager_reset_all(mock_config):
    """
    Verify reset_all clears elements from all queues.
    """
    qm = QueueManager(mock_config)
    qm.speech_queue.put("speech")
    assert qm.speech_queue.qsize() == 1
    
    qm.reset_all()
    assert qm.speech_queue.qsize() == 0

def test_queue_manager_occupancy_metrics(mock_config):
    """
    Verify occupancy metrics calculate sizes correctly.
    """
    qm = QueueManager(mock_config)
    
    qm.audio_queue.put(b"\x00")
    qm.audio_queue.put(b"\x00")
    
    metrics = qm.get_occupancy_metrics()
    assert metrics["audio_queue"]["size"] == 2
    assert metrics["audio_queue"]["maxsize"] == 10
    assert metrics["audio_queue"]["occupancy"] == 0.2
    assert metrics["speech_queue"]["occupancy"] == 0.0

def test_queue_manager_health_warnings(mock_config):
    """
    Verify health monitoring alerts on high occupancy and overflows.
    """
    qm = QueueManager(mock_config)
    
    # Healthy initially
    health = qm.monitor_health()
    assert health["speech_queue"] == "HEALTHY"
    
    # Fill speech_queue above 80% (9 items out of 10)
    for i in range(9):
        qm.speech_queue.put(f"item{i}")
        
    health = qm.monitor_health()
    assert health["speech_queue"] == "WARNING (OCCUPANCY >80%)"
    
    # Fill speech_queue completely
    qm.speech_queue.put("item9")
    assert qm.speech_queue.full() is True
    
    health = qm.monitor_health()
    assert health["speech_queue"] == "FULL (OVERFLOW RISK)"
