import pytest
import time
from stt.factory import STTFactory
from stt.providers.deepgram_streaming import DeepgramStreamingSTT
from stt.providers.assemblyai_streaming import AssemblyAIStreamingSTT
from stt.providers.parakeet_streaming import ParakeetStreamingSTT

@pytest.fixture
def base_config():
    return {
        "stt": {
            "deepgram": {"api_key": "MOCK_KEY"},
            "assemblyai": {"api_key": "MOCK_KEY"},
            "parakeet": {"url": "mock"}
        }
    }

def test_stt_provider_instantiation(base_config):
    """
    Verify factories return the correct concrete streaming STT adapters.
    """
    p1 = STTFactory.get_provider("deepgram_streaming", base_config)
    assert isinstance(p1, DeepgramStreamingSTT)
    
    p2 = STTFactory.get_provider("assemblyai_streaming", base_config)
    assert isinstance(p2, AssemblyAIStreamingSTT)
    
    p3 = STTFactory.get_provider("parakeet_streaming", base_config)
    assert isinstance(p3, ParakeetStreamingSTT)

def test_streaming_stt_callbacks(base_config):
    """
    Verify streaming STT providers invoke callbacks with expected outputs.
    """
    providers = ["deepgram_streaming", "assemblyai_streaming", "parakeet_streaming"]
    
    for prov_name in providers:
        prov = STTFactory.get_provider(prov_name, base_config)
        
        callback_data = []
        def on_transcript(text, is_final):
            callback_data.append((text, is_final))
            
        prov.start_stream("req-test-stt", on_transcript)
        prov.stream_audio(b"\x00\x00\x00")
        prov.stop_stream()
        
        # In mock mode, we expect at least the stop_stream final signal
        assert len(callback_data) > 0
        final_text, is_final = callback_data[-1]
        assert is_final is True
        assert "STT" in final_text or "Deepgram" in final_text or "Assembly" in final_text or "Parakeet" in final_text

def test_streaming_stt_batch_fallback(base_config):
    """
    Verify that the transcribe() method acts as a batch fallback.
    """
    providers = ["deepgram_streaming", "assemblyai_streaming", "parakeet_streaming"]
    mock_audio = b"\x00\x00"
    
    for prov_name in providers:
        prov = STTFactory.get_provider(prov_name, base_config)
        text = prov.transcribe(mock_audio)
        assert len(text) > 0
        assert "STT" in text or "Deepgram" in text or "Assembly" in text or "Parakeet" in text
