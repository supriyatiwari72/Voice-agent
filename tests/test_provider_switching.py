import pytest
from pipeline.pipeline_manager import PipelineManager
from workers.stt_worker import STTWorker
from workers.streaming_llm_worker import StreamingLLMWorker
from workers.streaming_tts_worker import StreamingTTSWorker
from stt.providers.deepgram_streaming import DeepgramStreamingSTT
from stt.providers.assemblyai_streaming import AssemblyAIStreamingSTT
from llm.providers.ollama_streaming import OllamaStreamingLLM
from llm.providers.nvidia_nim_streaming import NVIDIANIMStreamingLLM
from tts.providers.kokoro_streaming import KokoroStreamingTTS
from tts.providers.piper_streaming import PiperStreamingTTS

def test_config_driven_provider_switching():
    """
    Scenario 3: Verify that switching config active_providers maps
    correctly to provider instances inside worker components.
    """
    # Config Stack A
    config_a = {
        "active_providers": {
            "noise": "dummy",
            "vad": "dummy",
            "stt": "deepgram_streaming",
            "llm": "ollama_streaming",
            "tts": "kokoro_streaming"
        },
        "audio": {"buffer_max_size": 10},
        "queues": {
            "audio_queue_size": 10,
            "speech_queue_size": 10,
            "transcript_queue_size": 5,
            "response_queue_size": 5,
            "tts_queue_size": 5,
            "playback_queue_size": 5
        }
    }
    
    manager_a = PipelineManager(config_a)
    manager_a.initialize_pipeline()
    
    stt_worker_a = next(w for w in manager_a.workers if isinstance(w, STTWorker))
    llm_worker_a = next(w for w in manager_a.workers if isinstance(w, StreamingLLMWorker))
    tts_worker_a = next(w for w in manager_a.workers if isinstance(w, StreamingTTSWorker))
    
    assert isinstance(stt_worker_a.stt, DeepgramStreamingSTT)
    assert isinstance(llm_worker_a.llm, OllamaStreamingLLM)
    assert isinstance(tts_worker_a.tts, KokoroStreamingTTS)
    
    # Config Stack B
    config_b = {
        "active_providers": {
            "noise": "dummy",
            "vad": "dummy",
            "stt": "assemblyai_streaming",
            "llm": "nvidia_nim_streaming",
            "tts": "piper_streaming"
        },
        "audio": {"buffer_max_size": 10},
        "queues": {
            "audio_queue_size": 10,
            "speech_queue_size": 10,
            "transcript_queue_size": 5,
            "response_queue_size": 5,
            "tts_queue_size": 5,
            "playback_queue_size": 5
        }
    }
    
    manager_b = PipelineManager(config_b)
    manager_b.initialize_pipeline()
    
    stt_worker_b = next(w for w in manager_b.workers if isinstance(w, STTWorker))
    llm_worker_b = next(w for w in manager_b.workers if isinstance(w, StreamingLLMWorker))
    tts_worker_b = next(w for w in manager_b.workers if isinstance(w, StreamingTTSWorker))
    
    assert isinstance(stt_worker_b.stt, AssemblyAIStreamingSTT)
    assert isinstance(llm_worker_b.llm, NVIDIANIMStreamingLLM)
    assert isinstance(tts_worker_b.tts, PiperStreamingTTS)

def test_factory_invalid_provider_handling():
    """
    Verify that invalid provider names correctly raise ValueError exceptions.
    """
    from stt.factory import STTFactory
    from llm.factory import LLMFactory
    from tts.factory import TTSFactory
    
    with pytest.raises(ValueError):
        STTFactory.get_provider("invalid_stt")
        
    with pytest.raises(ValueError):
        LLMFactory.get_provider("invalid_llm")
        
    with pytest.raises(ValueError):
        TTSFactory.get_provider("invalid_tts")
