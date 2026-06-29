import pytest
from tts.factory import TTSFactory
from tts.providers.kokoro_streaming import KokoroStreamingTTS
from tts.providers.piper_streaming import PiperStreamingTTS

@pytest.fixture
def base_config():
    return {
        "tts": {
            "kokoro": {"voice": "af_bella"},
            "piper": {"model_path": "weights/en_US-lessac-medium.onnx"}
        }
    }

def test_tts_provider_instantiation(base_config):
    """
    Verify factories return the correct concrete streaming TTS adapters.
    """
    p1 = TTSFactory.get_provider("kokoro_streaming", base_config)
    assert isinstance(p1, KokoroStreamingTTS)
    
    p2 = TTSFactory.get_provider("piper_streaming", base_config)
    assert isinstance(p2, PiperStreamingTTS)

def test_streaming_tts_generation(base_config):
    """
    Verify streaming TTS providers yield audio chunks.
    """
    providers = ["kokoro_streaming", "piper_streaming"]
    
    for prov_name in providers:
        prov = TTSFactory.get_provider(prov_name, base_config)
        stream = prov.stream_synthesize("Hello world")
        
        chunks = list(stream)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0
            # PCM silence or simulated silence bytes
            assert b"\x00" in chunk

def test_streaming_tts_batch_fallback(base_config):
    """
    Verify that the synthesize() method acts as a batch fallback.
    """
    providers = ["kokoro_streaming", "piper_streaming"]
    
    for prov_name in providers:
        prov = TTSFactory.get_provider(prov_name, base_config)
        audio = prov.synthesize("hello")
        assert isinstance(audio, bytes)
        assert len(audio) > 0
        assert b"\x00" in audio
