"""
tests/test_stt_worker.py

Unit tests for the production STTWorker.
"""

import time
import pytest
from unittest.mock import MagicMock
from core.payloads import SpeechPayload, PartialTranscriptPayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from core.streaming_context import StreamingContext
from pipeline.pipeline_state import PipelineState
from workers.stt_worker import STTWorker


@pytest.fixture
def ctx_and_queues():
    config = {"queues": {"transcript_queue_size": 5, "partial_transcript_queue_size": 10}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    context.streaming_context = StreamingContext()
    return context, qm


def _make_worker(context, qm, transcribe_result="hello world"):
    mock_stt = MagicMock(spec=[])   # no attributes by default
    mock_stt.transcribe = MagicMock(return_value=transcribe_result)
    worker = STTWorker(
        context=context,
        input_queue=qm.transcript_queue,
        output_queue=qm.partial_transcript_queue,
        stt=mock_stt,
    )
    return worker, mock_stt


# ---------------------------------------------------------------------------
# 1. Batch transcription path
# ---------------------------------------------------------------------------

def test_batch_emits_single_final_payload(ctx_and_queues):
    """STTWorker emits exactly one PartialTranscriptPayload with is_final=True."""
    context, qm = ctx_and_queues
    worker, mock_stt = _make_worker(context, qm, "hello world")

    user_done = time.time() - 0.1
    payload = SpeechPayload(request_id="req-1", audio=b"\x00" * 960, user_done_timestamp=user_done)
    context.set_active_request_id("req-1")
    qm.transcript_queue.put(payload)

    worker.process_loop_step()

    assert qm.partial_transcript_queue.qsize() == 1
    out = qm.partial_transcript_queue.get()
    assert isinstance(out, PartialTranscriptPayload)
    assert out.text_chunk == "hello world"
    assert out.is_final is True
    assert out.request_id == "req-1"


def test_batch_records_stt_latency_metric(ctx_and_queues):
    """STTWorker records stt_latency_ms metric."""
    context, qm = ctx_and_queues
    worker, _ = _make_worker(context, qm, "test")
    context.set_active_request_id("req-m")

    payload = SpeechPayload(request_id="req-m", audio=b"\x00" * 960, user_done_timestamp=time.time())
    qm.transcript_queue.put(payload)
    worker.process_loop_step()

    summary = context.metrics.get_summary()
    assert summary["stt_latency_ms"]["count"] == 1


def test_batch_records_first_transcript_latency(ctx_and_queues):
    """STTWorker records first_partial_transcript_ms latency."""
    context, qm = ctx_and_queues
    worker, _ = _make_worker(context, qm, "hello")
    context.set_active_request_id("req-lat")

    payload = SpeechPayload(
        request_id="req-lat",
        audio=b"\x00" * 960,
        user_done_timestamp=time.time() - 0.2,
    )
    qm.transcript_queue.put(payload)
    worker.process_loop_step()

    summary = context.metrics.get_summary()
    assert summary["first_partial_transcript_ms"]["count"] == 1
    assert summary["first_partial_transcript_ms"]["average"] > 0


def test_batch_sets_transcribing_state(ctx_and_queues):
    """STTWorker transitions pipeline to TRANSCRIBING state."""
    context, qm = ctx_and_queues
    worker, _ = _make_worker(context, qm, "hi")
    context.set_active_request_id("req-state")

    payload = SpeechPayload(request_id="req-state", audio=b"\x00" * 960, user_done_timestamp=time.time())
    qm.transcript_queue.put(payload)
    worker.process_loop_step()

    assert qm.partial_transcript_queue.qsize() == 1


# ---------------------------------------------------------------------------
# 2. Stale / interrupted request handling
# ---------------------------------------------------------------------------

def test_stale_request_is_dropped(ctx_and_queues):
    """STTWorker drops payload when active_request_id differs."""
    context, qm = ctx_and_queues
    worker, mock_stt = _make_worker(context, qm, "ignored")
    context.set_active_request_id("req-other")  # different from payload's req-stale

    payload = SpeechPayload(request_id="req-stale", audio=b"\x00" * 960, user_done_timestamp=time.time())
    qm.transcript_queue.put(payload)
    worker.process_loop_step()

    mock_stt.transcribe.assert_not_called()
    assert qm.partial_transcript_queue.empty()


def test_interrupted_request_is_dropped(ctx_and_queues):
    """STTWorker drops payload when interruption_event is set."""
    context, qm = ctx_and_queues
    context.interruption_event.set()
    worker, mock_stt = _make_worker(context, qm, "ignored")
    context.set_active_request_id("req-int")

    payload = SpeechPayload(request_id="req-int", audio=b"\x00" * 960, user_done_timestamp=time.time())
    qm.transcript_queue.put(payload)
    worker.process_loop_step()

    mock_stt.transcribe.assert_not_called()
    assert qm.partial_transcript_queue.empty()


def test_interrupted_mid_transcription_result_dropped(ctx_and_queues):
    """If interrupted after transcription completes, the result is not forwarded."""
    context, qm = ctx_and_queues
    worker, mock_stt = _make_worker(context, qm, "result")

    def side_effect(audio):
        context.interruption_event.set()
        return "result"

    mock_stt.transcribe.side_effect = side_effect
    context.set_active_request_id("req-mid")

    payload = SpeechPayload(request_id="req-mid", audio=b"\x00" * 960, user_done_timestamp=time.time())
    qm.transcript_queue.put(payload)
    worker.process_loop_step()

    assert qm.partial_transcript_queue.empty()


# ---------------------------------------------------------------------------
# 3. Invalid payload handling
# ---------------------------------------------------------------------------

def test_invalid_payload_is_ignored(ctx_and_queues):
    """STTWorker gracefully ignores non-SpeechPayload inputs."""
    context, qm = ctx_and_queues
    worker, mock_stt = _make_worker(context, qm)

    qm.transcript_queue.put("not_a_payload")
    worker.process_loop_step()

    mock_stt.transcribe.assert_not_called()
    assert qm.partial_transcript_queue.empty()
