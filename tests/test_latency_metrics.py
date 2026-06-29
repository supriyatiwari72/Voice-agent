import os
import json
import pytest
import threading
from core.metrics import MetricsTracker

def test_metrics_tracker_initialization():
    """
    Verify all expected metric keys are initialized.
    """
    tracker = MetricsTracker()
    expected_keys = {
        "vad_latency_ms", "stt_latency_ms", "llm_latency_ms", "tts_latency_ms",
        "ttft_ms", "total_turnaround_ms", "first_partial_transcript_ms",
        "first_llm_token_ms", "first_sentence_ms", "first_audio_chunk_ms",
        "speech_end_ms", "memory_turn_count", "summary_count",
        "summary_generation_time_ms", "context_build_time_ms", "average_context_size",
        "wer_score", "speech_detection_latency_ms", "eos_detection_latency_ms",
        "playback_latency_ms"
    }
    assert set(tracker.metrics.keys()) == expected_keys
    for key in expected_keys:
        assert tracker.metrics[key] == []

def test_metrics_record_valid_key():
    """
    Verify recording valid metric records correctly.
    """
    tracker = MetricsTracker()
    tracker.record_metric("vad_latency_ms", 12.5)
    tracker.record_metric("vad_latency_ms", 15.0)
    assert tracker.metrics["vad_latency_ms"] == [12.5, 15.0]

def test_metrics_record_invalid_key():
    """
    Verify recording unregistered keys does not crash and gets ignored.
    """
    tracker = MetricsTracker()
    tracker.record_metric("invalid_key", 100.0)
    assert "invalid_key" not in tracker.metrics

def test_metrics_summary_empty():
    """
    Verify empty metrics summary returns zeroed structures.
    """
    tracker = MetricsTracker()
    summary = tracker.get_summary()
    assert summary["stt_latency_ms"] == {
        "average": 0.0,
        "min": 0.0,
        "max": 0.0,
        "count": 0.0
    }

def test_metrics_summary_averages():
    """
    Verify metrics summary calculates accurate average, min, max, count.
    """
    tracker = MetricsTracker()
    tracker.record_metric("ttft_ms", 10.0)
    tracker.record_metric("ttft_ms", 20.0)
    tracker.record_metric("ttft_ms", 30.0)
    
    summary = tracker.get_summary()
    assert summary["ttft_ms"] == {
        "average": 20.0,
        "min": 10.0,
        "max": 30.0,
        "count": 3.0
    }

def test_metrics_export_json_valid_path(tmp_path):
    """
    Verify JSON export writes correct schema to a file path.
    """
    tracker = MetricsTracker()
    tracker.record_metric("tts_latency_ms", 5.0)
    
    export_path = os.path.join(tmp_path, "test_metrics.json")
    tracker.export_json(export_path)
    
    assert os.path.exists(export_path) is True
    with open(export_path, "r") as f:
        data = json.load(f)
        
    assert "summary" in data
    assert "raw" in data
    assert data["summary"]["tts_latency_ms"] == 5.0
    assert data["raw"]["tts_latency_ms"] == [5.0]

def test_metrics_export_json_default_path():
    """
    Verify default export path works and creates metrics.json.
    """
    tracker = MetricsTracker()
    tracker.record_metric("total_turnaround_ms", 450.0)
    tracker.export_json()
    assert os.path.exists("metrics.json") is True

def test_metrics_thread_safety():
    """
    Verify thread-safety when recording metrics from multiple concurrent threads.
    """
    tracker = MetricsTracker()
    num_threads = 20
    writes_per_thread = 50
    
    threads = []
    def worker():
        for _ in range(writes_per_thread):
            tracker.record_metric("llm_latency_ms", 2.0)
            
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(tracker.metrics["llm_latency_ms"]) == num_threads * writes_per_thread
