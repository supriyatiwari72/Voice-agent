import pytest
import queue
import time
from unittest.mock import MagicMock
from core.payloads import AudioPayload, SpeechPayload
from core.pipeline_context import PipelineContext
from core.metrics import MetricsTracker
from core.queue_manager import QueueManager
from pipeline.pipeline_state import PipelineState
from workers.vad_worker import VADWorker

@pytest.fixture
def mock_context_and_queues():
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 1,
                    "min_speech_bytes": 0
                }
            }
        }
    }
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    context.ptt_active.set()
    return context, qm

def test_vad_worker_speech_accumulation(mock_context_and_queues):
    """
    Verify VADWorker accumulates speech chunks when VAD detects active voice,
    and forwards a complete SpeechPayload upon transitioning to silence.
    """
    context, qm = mock_context_and_queues
    
    mock_vad = MagicMock()
    worker = VADWorker(
        context=context,
        input_queue=qm.speech_queue,
        output_queue=qm.transcript_queue,
        vad=mock_vad
    )
    
    # 1. First payload chunk: VAD detects speech
    mock_vad.is_speech.return_value = True
    payload1 = AudioPayload(request_id="req-vad", audio=b"\x01\x02", created_at=time.time())
    qm.speech_queue.put(payload1)
    
    worker.process_loop_step()
    
    # Should transition to LISTENING and accumulate, output remains empty
    assert context.get_state() == PipelineState.LISTENING
    assert worker._speech_buffer == b"\x01\x02"
    assert qm.transcript_queue.empty() is True
    
    # 2. Second payload chunk: VAD still detects speech
    payload2 = AudioPayload(request_id="req-vad", audio=b"\x03\x04", created_at=time.time())
    qm.speech_queue.put(payload2)
    
    worker.process_loop_step()
    assert worker._speech_buffer == b"\x01\x02\x03\x04"
    assert qm.transcript_queue.empty() is True

    # 3. Third payload chunk: VAD detects silence (speech end boundary)
    mock_vad.is_speech.return_value = False
    payload3 = AudioPayload(request_id="req-vad", audio=b"\x00\x00", created_at=time.time())
    qm.speech_queue.put(payload3)
    
    worker.process_loop_step()
    
    # Context should transition to PROCESSING, and output queue holds the complete segment
    assert context.get_state() == PipelineState.PROCESSING
    assert qm.transcript_queue.qsize() == 1
    
    out_payload = qm.transcript_queue.get()
    assert isinstance(out_payload, SpeechPayload)
    assert out_payload.audio == b"\x01\x02\x03\x04"
    assert out_payload.request_id == "req-vad"
    assert out_payload.user_done_timestamp > 0.0
    
    # State reset verified
    assert worker._in_speech is False
    assert worker._speech_buffer == b""


def test_vad_single_pause_does_not_terminate():
    """Verify that a single pause below the silence threshold does not terminate the turn."""
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 3,
                    "min_speech_bytes": 8000
                }
            }
        }
    }
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    context.ptt_active.set()
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # 1. Speech frame
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-1", b"\x01" * 1000, time.time()))
    assert worker._in_speech is True
    assert worker._silence_frames == 0
    assert context.queue_manager.transcript_queue.empty() is True

    # 2. Silence frame (1st pause)
    mock_vad.is_speech.return_value = False
    worker.process(AudioPayload("req-1", b"\x00" * 1000, time.time()))
    assert worker._in_speech is True
    assert worker._silence_frames == 1
    assert context.queue_manager.transcript_queue.empty() is True

    # 3. Speech resumes
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-1", b"\x01" * 1000, time.time()))
    assert worker._in_speech is True
    assert worker._silence_frames == 0  # Counter reset
    assert context.queue_manager.transcript_queue.empty() is True


def test_vad_multiple_pauses_below_threshold():
    """Verify multiple pauses below the max_silence_frames threshold do not terminate speech."""
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 5,
                    "min_speech_bytes": 8000
                }
            }
        }
    }
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    context.ptt_active.set()
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # Speech starts
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-2", b"\x01" * 1000, time.time()))

    # 4 consecutive silent frames (threshold is 5)
    mock_vad.is_speech.return_value = False
    for i in range(4):
        worker.process(AudioPayload("req-2", b"\x00" * 1000, time.time()))
        assert worker._in_speech is True
        assert worker._silence_frames == i + 1

    # Speech resumes on 5th frame
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-2", b"\x01" * 1000, time.time()))
    assert worker._in_speech is True
    assert worker._silence_frames == 0  # Reset
    assert context.queue_manager.transcript_queue.empty() is True


def test_vad_silence_threshold_finalizes_correctly():
    """Verify that reaching the silence threshold finalizes speech and forwards if long enough."""
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 3,
                    "min_speech_bytes": 3000
                }
            }
        }
    }
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    context.ptt_active.set()
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # 1. Speech frame (3000 bytes)
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-3", b"\x01" * 3000, time.time()))

    # 2. Silence frames (3 frames of 1000 bytes each)
    mock_vad.is_speech.return_value = False
    for _ in range(3):
        worker.process(AudioPayload("req-3", b"\x00" * 1000, time.time()))

    # Reached silence threshold (3 frames) -> Speech finalized
    # Actual speech is 3000 bytes (after stripping 3000 bytes of silence)
    # Since 3000 >= 3000 (min_speech_bytes), it should be forwarded!
    assert context.queue_manager.transcript_queue.qsize() == 1
    speech = context.queue_manager.transcript_queue.get()
    assert isinstance(speech, SpeechPayload)
    assert len(speech.audio) == 3000
    assert worker._in_speech is False
    assert worker._silence_frames == 0


def test_vad_short_clicks_are_discarded():
    """Verify that short clicks/noise (< min_speech_bytes) are finalized but discarded cleanly."""
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 2,
                    "min_speech_bytes": 8000
                }
            }
        }
    }
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    context.ptt_active.set()
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # 1. Speech frame (1000 bytes)
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-4", b"\x01" * 1000, time.time()))

    # 2. Silence frames (2 frames of 1000 bytes)
    mock_vad.is_speech.return_value = False
    worker.process(AudioPayload("req-4", b"\x00" * 1000, time.time()))
    worker.process(AudioPayload("req-4", b"\x00" * 1000, time.time()))

    # Finalized turn. Total accumulated: 3000 bytes.
    # Since 3000 < 8000, it should be discarded cleanly and NOT forwarded.
    assert context.queue_manager.transcript_queue.empty() is True
    assert worker._in_speech is False
    assert worker._speech_buffer == b""


def test_vad_short_commands_are_preserved():
    """Verify that short commands (e.g. "yes", "no" >= 8000 bytes) are preserved and forwarded."""
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 2,
                    "min_speech_bytes": 8000
                }
            }
        }
    }
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    context.ptt_active.set()
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # 1. Speech frame representing "yes" (8000 bytes)
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-5", b"\x01" * 8000, time.time()))

    # 2. Silence frames to finalize (2 frames of 1500 bytes each)
    mock_vad.is_speech.return_value = False
    worker.process(AudioPayload("req-5", b"\x00" * 1500, time.time()))
    worker.process(AudioPayload("req-5", b"\x00" * 1500, time.time()))

    # Finalized. Actual speech: 8000 bytes.
    # Since 8000 >= 8000, it is preserved and forwarded.
    assert context.queue_manager.transcript_queue.qsize() == 1
    speech = context.queue_manager.transcript_queue.get()
    assert speech.audio == (b"\x01" * 8000)


def test_vad_interruption_barge_in_handling():
    """Verify that VADWorker triggers interruption/barge-in when speech starts in SPEAKING state."""
    config = {
        "queues": {
            "speech_queue_size": 5,
            "transcript_queue_size": 5,
            "interruption_queue": 5
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
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # Set state to SPEAKING (agent is speaking)
    context.set_state(PipelineState.SPEAKING)

    # User barge-in: VAD detects speech
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-barge", b"\x7f" * 1000, time.time()))

    # Interruption event should be set
    assert context.interruption_event.is_set() is True
    # Interruption queue should have the payload
    assert context.queue_manager.interruption_queue.qsize() == 1
    barge = context.queue_manager.interruption_queue.get()
    assert barge.request_id == "req-barge"


def test_vad_push_to_talk_ignored_accumulation():
    """Verify that when ptt_active is not set, speech frames are not accumulated and not finalized."""
    config = {
        "queues": {"speech_queue_size": 5, "transcript_queue_size": 5},
        "models_meta": {
            "vad_providers": {
                "silero": {
                    "max_silence_frames": 2,
                    "min_speech_bytes": 0
                }
            }
        }
    }
    context = PipelineContext(config, QueueManager(config), MetricsTracker())
    context.set_state(PipelineState.IDLE)
    # Ensure ptt_active is NOT set (simulating no button click)
    context.ptt_active.clear()
    
    mock_vad = MagicMock()
    worker = VADWorker(context, context.queue_manager.speech_queue, context.queue_manager.transcript_queue, mock_vad)

    # 1. User speaks: VAD detects speech
    mock_vad.is_speech.return_value = True
    worker.process(AudioPayload("req-ptt-ignore", b"\x01" * 2000, time.time()))

    # Should NOT start speech segment
    assert worker._in_speech is False
    assert len(worker._speech_buffer) == 0

    # 2. Silence frame
    mock_vad.is_speech.return_value = False
    worker.process(AudioPayload("req-ptt-ignore", b"\x00" * 2000, time.time()))

    # Should NOT finalize or forward anything to transcript queue
    assert context.queue_manager.transcript_queue.empty() is True
    assert worker._in_speech is False


