import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from stt.factory import STTFactory
from stt.providers.distil_whisper import DistilWhisperSTT
from stt.base import BaseSTT

class MockSegment:
    def __init__(self, text: str):
        self.text = text

@pytest.fixture
def mock_distil_config():
    return {
        "audio": {
            "sample_rate": 16000
        },
        "models_meta": {
            "stt_providers": {
                "distil_whisper": {
                    "model_size": "distil-large-v3",
                    "device": "cpu",
                    "compute_type": "int8",
                    "beam_size": 5
                }
            }
        }
    }

@patch("stt.providers.distil_whisper.WhisperModel")
def test_distil_whisper_initialization(mock_whisper_cls, mock_distil_config):
    """
    Verify that DistilWhisperSTT initializes and warms up correctly.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([MockSegment("warmup")], None)

    stt = DistilWhisperSTT(mock_distil_config)
    assert stt.model is not None
    assert stt.beam_size == 5
    
    # Assert model instantiation values from configuration
    mock_whisper_cls.assert_called_once_with("distil-large-v3", device="cpu", compute_type="int8")
    assert mock_instance.transcribe.call_count == 1

@patch("stt.providers.distil_whisper.WhisperModel")
def test_distil_whisper_factory_creation(mock_whisper_cls, mock_distil_config):
    """
    Verify factory resolves and instantiates DistilWhisperSTT correctly.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    provider = STTFactory.get_provider("distil_whisper", mock_distil_config)
    assert isinstance(provider, DistilWhisperSTT)
    assert isinstance(provider, BaseSTT)

@patch("stt.providers.distil_whisper.WhisperModel")
def test_distil_whisper_transcribe(mock_whisper_cls, mock_distil_config):
    """
    Verify transcribe() correctly outputs transcription string.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([MockSegment("Hello from Distil Whisper")], None)

    stt = DistilWhisperSTT(mock_distil_config)
    result = stt.transcribe(b"\x00" * 1024)
    
    assert isinstance(result, str)
    assert result == "Hello from Distil Whisper"

@patch("stt.providers.distil_whisper.WhisperModel")
def test_distil_whisper_handles_empty_audio(mock_whisper_cls, mock_distil_config):
    """
    Verify transcribe() handles empty bytes gracefully.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    stt = DistilWhisperSTT(mock_distil_config)
    assert stt.transcribe(b"") == ""
    assert stt.transcribe(None) == ""

@patch("stt.providers.distil_whisper.WhisperModel")
def test_distil_whisper_handles_invalid_audio(mock_whisper_cls, mock_distil_config):
    """
    Verify transcribe() handles invalid or odd length bytes gracefully.
    """
    mock_instance = MagicMock()
    mock_whisper_cls.return_value = mock_instance
    mock_instance.transcribe.return_value = ([], None)

    stt = DistilWhisperSTT(mock_distil_config)
    assert stt.transcribe(b"\x00") == ""

@patch("stt.providers.distil_whisper.WhisperModel")
def test_distil_whisper_handles_invalid_model(mock_whisper_cls, mock_distil_config):
    """
    Verify that if WhisperModel initialization fails, it handles it gracefully.
    """
    mock_whisper_cls.side_effect = ValueError("Invalid model size or device configuration")

    stt = DistilWhisperSTT(mock_distil_config)
    assert stt.model is None
    # transcription should return empty string without crashing
    assert stt.transcribe(b"\x00" * 1024) == ""
