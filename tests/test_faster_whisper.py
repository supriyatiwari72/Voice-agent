import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from stt.factory import STTFactory
from stt.providers.faster_whisper import FasterWhisperSTT
from stt.base import BaseSTT

# Mock segment helper to simulate Faster Whisper segments iterator
class MockSegment:
    def __init__(self, text: str):
        self.text = text

@pytest.fixture
def mock_whisper_config():
    return {
        "audio": {
            "sample_rate": 16000
        },
        "models_meta": {
            "stt_providers": {
                "faster_whisper": {
                    "model_size": "tiny",
                    "device": "cpu",
                    "compute_type": "int8",
                    "beam_size": 5
                }
            }
        }
    }

@patch("stt.providers.faster_whisper.WhisperModel")
def test_faster_whisper_model_initialization(mock_whisper_cls, mock_whisper_config):
    """
    Test 1: Model initialization and warm-up succeeds.
    """
    # Configure mock WhisperModel instance
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([MockSegment("warmup")], None)

    stt = FasterWhisperSTT(mock_whisper_config)
    assert stt.model is not None
    assert stt.beam_size == 5
    
    # Check that WhisperModel was constructed and warmed up
    mock_whisper_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")
    assert mock_instance.transcribe.call_count == 1

@patch("stt.providers.faster_whisper.WhisperModel")
def test_faster_whisper_factory_creation(mock_whisper_cls, mock_whisper_config):
    """
    Test 2: Factory creates FasterWhisperSTT correctly.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    provider = STTFactory.get_provider("faster_whisper", mock_whisper_config)
    assert isinstance(provider, FasterWhisperSTT)
    assert isinstance(provider, BaseSTT)

@patch("stt.providers.faster_whisper.WhisperModel")
def test_faster_whisper_returns_string(mock_whisper_cls, mock_whisper_config):
    """
    Test 3: transcribe() returns a string.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([MockSegment("What is artificial intelligence?")], None)

    stt = FasterWhisperSTT(mock_whisper_config)
    
    # Input PCM data (512 samples = 1024 bytes)
    dummy_pcm = b"\x00" * 1024
    result = stt.transcribe(dummy_pcm)
    
    assert isinstance(result, str)
    assert result == "What is artificial intelligence?"

@patch("stt.providers.faster_whisper.WhisperModel")
def test_faster_whisper_handles_empty_audio(mock_whisper_cls, mock_whisper_config):
    """
    Test 4: transcribe() handles empty bytes safely, returning empty string.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    stt = FasterWhisperSTT(mock_whisper_config)
    
    assert stt.transcribe(b"") == ""
    assert stt.transcribe(None) == ""

@patch("stt.providers.faster_whisper.WhisperModel")
def test_faster_whisper_handles_invalid_audio(mock_whisper_cls, mock_whisper_config):
    """
    Test 5: transcribe() handles invalid or odd byte arrays safely without crashing.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    stt = FasterWhisperSTT(mock_whisper_config)
    
    # Odd byte sequence (invalid PCM)
    assert stt.transcribe(b"\x00") == ""

@patch("stt.providers.faster_whisper.WhisperModel")
def test_faster_whisper_provider_switching(mock_whisper_cls, mock_whisper_config):
    """
    Test 6: Provider switching resolves correct types dynamically via Factory.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    dummy_stt = STTFactory.get_provider("dummy", mock_whisper_config)
    assert dummy_stt is not None
    
    fw_stt = STTFactory.get_provider("faster_whisper", mock_whisper_config)
    assert fw_stt is not None
    assert fw_stt != dummy_stt

@patch("stt.providers.faster_whisper.WhisperModel")
def test_pipeline_independence_from_stt_provider(mock_whisper_cls, mock_whisper_config):
    """
    Test 7: Pipeline remains fully independent of concrete STT provider type.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([MockSegment("Hi")], None)

    from pipeline.pipeline_manager import PipelineManager
    from workers.stt_worker import STTWorker

    # Build config utilizing dummy/mock providers to prevent heavy downloads/API calls
    config = {
        "active_providers": {
            "noise": "dummy",
            "vad": "dummy",
            "stt": "faster_whisper",
            "llm": "dummy",
            "tts": "dummy"
        },
        "models_meta": mock_whisper_config["models_meta"]
    }

    manager = PipelineManager(config)
    manager.initialize_pipeline()

    # Retrieve STT worker from manager and verify the STT provider resolves as BaseSTT
    stt_worker = next(w for w in manager.workers if isinstance(w, STTWorker))
    assert isinstance(stt_worker.stt, BaseSTT)
    assert stt_worker.stt.__class__.__name__ == "FasterWhisperSTT"

