import pytest
import queue
import time
from core.payloads import PartialResponsePayload, SentencePayload
from core.pipeline_context import PipelineContext
from core.queue_manager import QueueManager
from core.metrics import MetricsTracker
from core.streaming_context import StreamingContext
from workers.sentence_aggregator_worker import SentenceAggregatorWorker

@pytest.fixture
def mock_context_and_queues():
    config = {"queues": {"partial_response_queue_size": 5, "tts_queue_size": 5}}
    metrics = MetricsTracker()
    qm = QueueManager(config)
    context = PipelineContext(config, qm, metrics)
    context.streaming_context = StreamingContext()
    return context, qm

def test_sentence_aggregation_logic(mock_context_and_queues):
    """
    Verify that SentenceAggregatorWorker groups tokens into complete sentence payloads.
    """
    context, qm = mock_context_and_queues
    
    worker = SentenceAggregatorWorker(
        context=context,
        input_queue=qm.partial_response_queue,
        output_queue=qm.tts_queue
    )
    
    context.set_active_request_id("req-agg")
    
    # 1. Send first word: no sentence boundary
    qm.partial_response_queue.put(PartialResponsePayload("req-agg", "Hello", is_final=False, timestamp=time.time()))
    worker.process_loop_step()
    assert qm.tts_queue.empty() is True
    
    # 2. Send sentence punctuation boundary
    qm.partial_response_queue.put(PartialResponsePayload("req-agg", ", how are you?", is_final=False, timestamp=time.time()))
    worker.process_loop_step()
    
    # Punctuation boundaries split the text into two sentences: "Hello," and "how are you?"
    assert qm.tts_queue.qsize() == 2
    s1 = qm.tts_queue.get()
    assert isinstance(s1, SentencePayload)
    assert s1.text == "Hello,"
    assert s1.is_final is False

    s2 = qm.tts_queue.get()
    assert isinstance(s2, SentencePayload)
    assert s2.text == "how are you?"
    assert s2.is_final is False
    
    # 3. Send final word and mark is_final = True (forcing final flush)
    qm.partial_response_queue.put(PartialResponsePayload("req-agg", " Fine", is_final=True, timestamp=time.time()))
    worker.process_loop_step()
    
    # Verify final flush
    assert qm.tts_queue.qsize() == 1
    s2 = qm.tts_queue.get()
    assert s2.text == "Fine"
    assert s2.is_final is True
    
    # Verify first_sentence_ms telemetry
    summary = context.metrics.get_summary()
    assert summary["first_sentence_ms"]["count"] == 1.0

def test_aggregator_token_accumulation(mock_context_and_queues):
    """
    Verify that SentenceAggregatorWorker aggregates incoming tokens ("Hello", ",", "how", ...)
    and dispatches complete SentencePayload segments instead of raw individual tokens.
    """
    context, qm = mock_context_and_queues
    worker = SentenceAggregatorWorker(
        context=context,
        input_queue=qm.partial_response_queue,
        output_queue=qm.tts_queue
    )
    context.set_active_request_id("req-accum")
    
    tokens = ["Hello", ",", " how", " are", " you", " today", "?"]
    for i, tok in enumerate(tokens):
        is_final = (i == len(tokens) - 1)
        payload = PartialResponsePayload("req-accum", tok, is_final=is_final, timestamp=time.time())
        qm.partial_response_queue.put(payload)
        worker.process_loop_step()
        
    # We expect exactly 3 SentencePayload dispatches:
    # 1. "Hello," (triggered by comma)
    # 2. "how are you today?" (triggered by question mark)
    # 3. "" (triggered by final propagation)
    assert qm.tts_queue.qsize() == 3
    s1 = qm.tts_queue.get()
    assert s1.text == "Hello,"
    assert s1.is_final is False
    
    s2 = qm.tts_queue.get()
    assert s2.text == "how are you today?"
    assert s2.is_final is False

    s3 = qm.tts_queue.get()
    assert s3.text == ""
    assert s3.is_final is True

def test_sentence_aggregator_long_response(mock_context_and_queues):
    """
    Stress test with 1000+ tokens to verify no aggregator backlog or memory leaks.
    """
    context, qm = mock_context_and_queues
    worker = SentenceAggregatorWorker(
        context=context,
        input_queue=qm.partial_response_queue,
        output_queue=qm.tts_queue
    )
    context.set_active_request_id("req-long")
    
    # Send 1050 tokens (e.g. 50 sentences of 20 words each, ending with periods)
    num_sentences = 50
    words_per_sentence = 20
    
    start_time = time.time()
    dispatched_count = 0
    for s_idx in range(num_sentences):
        for w_idx in range(words_per_sentence):
            token = "word "
            qm.partial_response_queue.put(PartialResponsePayload("req-long", token, is_final=False, timestamp=start_time))
            worker.process_loop_step()
        # End of sentence
        qm.partial_response_queue.put(PartialResponsePayload("req-long", ".", is_final=False, timestamp=start_time))
        worker.process_loop_step()
        
        # Pop from tts_queue to prevent deadlock (max queue capacity is 5)
        while not qm.tts_queue.empty():
            s = qm.tts_queue.get()
            dispatched_count += 1
            assert s.text.startswith("word")
            assert s.is_final is False
        
    # Send final flushing payload
    qm.partial_response_queue.put(PartialResponsePayload("req-long", " final sentence.", is_final=True, timestamp=start_time))
    worker.process_loop_step()
    
    # Pop remaining
    while not qm.tts_queue.empty():
        s = qm.tts_queue.get()
        dispatched_count += 1
        if s.text == "final sentence.":
            assert s.is_final is False
        elif s.text == "":
            assert s.is_final is True
    
    # We expect 52 sentence payloads in total (50 standard + 1 final text + 1 empty final propagation)
    assert dispatched_count == 52
    # Check that aggregator's internal buffer is empty
    assert worker._buffer == ""
