import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from tts.factory import TTSFactory
from tts.providers.kokoro import KokoroTTS
from tts.base import BaseTTS

@pytest.fixture
def mock_kokoro_config():
    return {
        "models_meta": {
            "tts_providers": {
                "kokoro": {
                    "model_path": "weights/kokoro-v0_19.onnx",
                    "voices_path": "weights/voices.bin",
                    "voice": "af_bella"
                }
            }
        }
    }

def test_kokoro_factory_creation(mock_kokoro_config):
    """
    Verify that TTSFactory correctly instantiates KokoroTTS.
    """
    # Force it into fallback mode for base testing if weights do not exist
    with patch("os.path.exists", return_value=False):
        provider = TTSFactory.get_provider("kokoro", mock_kokoro_config)
        assert isinstance(provider, KokoroTTS)
        assert isinstance(provider, BaseTTS)
        assert provider.fallback is True

def test_kokoro_missing_files_graceful_fallback(mock_kokoro_config):
    """
    Verify that if the weight files do not exist, KokoroTTS sets 
    fallback to True and does not crash during initialization.
    """
    with patch("os.path.exists", return_value=False):
        tts = KokoroTTS(mock_kokoro_config)
        assert tts.fallback is True
        # Verify it returns dummy silence instead of crashing
        res = tts.synthesize("Hello")
        assert len(res) == 32000  # 1 second of 16kHz mono 16-bit PCM silence

def test_kokoro_synthesize_success(mock_kokoro_config):
    """
    Verify that float32 waveform numpy arrays are correctly 
    converted/normalized into int16 PCM bytes.
    """
    with patch("os.path.exists", return_value=True), \
         patch("tts.providers.kokoro.KOKORO_AVAILABLE", True), \
         patch("tts.providers.kokoro.Kokoro") as mock_kokoro_cls:
        
        mock_instance = MagicMock()
        mock_kokoro_cls.return_value = mock_instance
        # Mock Kokoro.create output: float32 numpy array and sample rate
        mock_waveform = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        mock_instance.create.return_value = (mock_waveform, 24000)

        tts = KokoroTTS(mock_kokoro_config)
        assert tts.fallback is False
        
        result = tts.synthesize("Hello world")
        
        # Verify the float-to-int16 mapping: 
        # 0.0 -> 0
        # 0.5 -> 16383
        # -0.5 -> -16383
        # 1.0 -> 32767
        # -1.0 -> -32767
        expected_array = np.array([0, 16383, -16383, 32767, -32767], dtype=np.int16)
        assert result == expected_array.tobytes()

def test_kokoro_synthesize_stream(mock_kokoro_config):
    """
    Verify that synthesize_stream splits input text by sentence and yields chunks.
    """
    with patch("os.path.exists", return_value=True), \
         patch("tts.providers.kokoro.KOKORO_AVAILABLE", True), \
         patch("tts.providers.kokoro.Kokoro") as mock_kokoro_cls:
        
        mock_instance = MagicMock()
        mock_kokoro_cls.return_value = mock_instance
        # Yield a 2-sample float array for each sentence
        mock_instance.create.return_value = (np.array([0.1, -0.1], dtype=np.float32), 24000)

        tts = KokoroTTS(mock_kokoro_config)
        
        stream = tts.synthesize_stream("Sentence one. Sentence two! Sentence three?")
        chunks = list(stream)
        
        # There should be exactly 3 chunks generated (one for each split sentence)
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) == 4  # 2 samples of int16 = 4 bytes

def test_kokoro_invalid_voice_fallback(mock_kokoro_config):
    """
    Verify that if an invalid voice is specified, KokoroTTS logs a warning 
    and falls back to the default 'af_bella' voice.
    """
    config = {
        "models_meta": {
            "tts_providers": {
                "kokoro": {
                    "model_path": "weights/kokoro-v0_19.onnx",
                    "voices_path": "weights/voices.bin",
                    "voice": "invalid_voice_name"
                }
            }
        }
    }
    with patch("os.path.exists", return_value=True), \
         patch("tts.providers.kokoro.KOKORO_AVAILABLE", True), \
         patch("tts.providers.kokoro.Kokoro") as mock_kokoro_cls:
        
        mock_instance = MagicMock()
        mock_kokoro_cls.return_value = mock_instance
        
        tts = KokoroTTS(config)
        assert tts.voice == "af_bella"
