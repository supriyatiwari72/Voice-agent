"""
tests/test_audio_recorder.py

Unit tests for the production AudioRecorder.

sounddevice is fully mocked so these tests run without any hardware or
audio driver being present (CI-safe).
"""

import time
import threading
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from audio.audio_buffer import AudioBuffer
from audio.recorder import AudioRecorder

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "frame_duration_ms": 30,
        "input_device": None,
    }
}


def _make_recorder(config=None):
    buf = AudioBuffer(max_size=200)
    rec = AudioRecorder(config or BASE_CONFIG, buf)
    return rec, buf


def _make_float32_frame(samples=480, amplitude=0.5):
    """Return a float32 numpy array simulating one mic frame."""
    rng = np.random.default_rng(42)
    data = (rng.random(samples) * 2 - 1) * amplitude
    return data.astype(np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

def test_config_defaults_applied():
    """Recorder correctly parses sample_rate, channels, blocksize from config."""
    rec, _ = _make_recorder()
    assert rec._sample_rate == 16000
    assert rec._channels == 1
    assert rec._frame_ms == 30
    assert rec._blocksize == 480          # 16000 * 0.030
    assert rec._device is None


def test_config_custom_device():
    """input_device is forwarded to the recorder."""
    cfg = {"audio": {"sample_rate": 16000, "channels": 1, "frame_duration_ms": 30, "input_device": 2}}
    rec, _ = _make_recorder(cfg)
    assert rec._device == 2


# ---------------------------------------------------------------------------
# 2. start_recording / stop_recording
# ---------------------------------------------------------------------------

def test_start_recording_opens_stream(capsys):
    """start_recording() opens a sounddevice InputStream and prints log lines."""
    rec, _ = _make_recorder()

    mock_stream = MagicMock()
    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        rec.start_recording()

    assert rec.is_active()
    mock_sd.InputStream.assert_called_once()
    mock_stream.start.assert_called_once()

    captured = capsys.readouterr()
    assert "Recorder Started" in captured.out
    assert "Microphone Opened" in captured.out

    rec._active = False  # cleanup without calling stop (stream is a mock)


def test_start_recording_idempotent():
    """Calling start_recording() twice does not open a second stream."""
    rec, _ = _make_recorder()

    mock_stream = MagicMock()
    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        rec.start_recording()
        rec.start_recording()   # second call — should be a no-op

    assert mock_sd.InputStream.call_count == 1
    rec._active = False


def test_stop_recording_closes_stream(capsys):
    """stop_recording() calls stream.stop() and stream.close()."""
    rec, _ = _make_recorder()

    mock_stream = MagicMock()
    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        rec.start_recording()
        rec.stop_recording()

    assert not rec.is_active()
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()

    captured = capsys.readouterr()
    assert "Recorder Stopped" in captured.out


def test_stop_recording_when_not_started():
    """stop_recording() is a no-op when never started — no exception raised."""
    rec, _ = _make_recorder()
    rec.stop_recording()  # must not raise
    assert not rec.is_active()


def test_is_active_reflects_state():
    """is_active() returns False before start and True after start."""
    rec, _ = _make_recorder()
    assert not rec.is_active()

    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = MagicMock()

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        rec.start_recording()
        assert rec.is_active()

    rec._active = False


# ---------------------------------------------------------------------------
# 3. Audio callback — PCM conversion and buffer push
# ---------------------------------------------------------------------------

def test_audio_callback_pushes_pcm_to_buffer():
    """_audio_callback converts float32 frame to int16 PCM and pushes it."""
    rec, buf = _make_recorder()
    rec._active = True

    frame = _make_float32_frame(480)
    rec._audio_callback(frame, 480, None, None)

    assert buf.size() == 1
    pcm_bytes = buf.pop(timeout=0.1)
    assert pcm_bytes is not None
    assert len(pcm_bytes) == 960           # 480 samples × 2 bytes/sample


def test_audio_callback_pcm_values_in_range():
    """PCM values are clipped correctly — no int16 overflow, pre-emphasis applied."""
    rec, buf = _make_recorder()
    rec._active = True

    saturated = np.ones((480, 1), dtype=np.float32)
    rec._audio_callback(saturated, 480, None, None)

    pcm_bytes = buf.pop(timeout=0.1)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert np.all(samples <= 32767)
    assert np.all(samples >= -32768)
    assert samples[0] == 32767


def test_audio_callback_negative_saturation():
    """Negative saturated input clips to -32767 (not -32768 due to float conversion)."""
    rec, buf = _make_recorder()
    rec._active = True

    saturated = np.full((480, 1), -1.0, dtype=np.float32)
    rec._audio_callback(saturated, 480, None, None)

    pcm_bytes = buf.pop(timeout=0.1)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert np.all(samples >= -32768)
    assert np.all(samples <= 32767)
    assert samples[0] == -32767


def test_audio_callback_dropped_when_inactive():
    """_audio_callback is a no-op when recorder is not active."""
    rec, buf = _make_recorder()
    rec._active = False

    frame = _make_float32_frame(480)
    rec._audio_callback(frame, 480, None, None)

    assert buf.size() == 0


def test_audio_callback_buffer_full_does_not_raise():
    """Callback gracefully drops frames when the buffer is full."""
    rec, buf = _make_recorder({"audio": {"sample_rate": 16000, "channels": 1,
                                          "frame_duration_ms": 30, "input_device": None}})
    # Override buffer with a size-1 buffer and pre-fill it
    small_buf = AudioBuffer(max_size=1)
    small_buf.push(b"\x00" * 960)
    rec.input_buffer = small_buf
    rec._active = True

    frame = _make_float32_frame(480)
    rec._audio_callback(frame, 480, None, None)  # must not raise

    assert small_buf.size() == 1   # still only 1 item — dropped, not appended


# ---------------------------------------------------------------------------
# 4. Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_callbacks_are_thread_safe():
    """Multiple threads calling _audio_callback concurrently do not corrupt the buffer."""
    rec, buf = _make_recorder()
    rec._active = True

    frame = _make_float32_frame(480)
    errors = []

    def push_frames():
        try:
            for _ in range(20):
                rec._audio_callback(frame.copy(), 480, None, None)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=push_frames) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety errors: {errors}"
    # At most 4×20 = 80 frames pushed into a 200-slot buffer
    assert buf.size() <= 80


# ---------------------------------------------------------------------------
# 5. Missing sounddevice raises RuntimeError
# ---------------------------------------------------------------------------

def test_missing_sounddevice_raises_runtime_error():
    """start_recording() raises RuntimeError when sounddevice is not installed."""
    rec, _ = _make_recorder()

    with patch.dict("sys.modules", {"sounddevice": None}):
        with pytest.raises((RuntimeError, ImportError)):
            rec.start_recording()


# ---------------------------------------------------------------------------
# 6. InputStream receives correct parameters
# ---------------------------------------------------------------------------

def test_stream_opened_with_correct_params():
    """InputStream is constructed with the exact sample_rate, blocksize, and device from config."""
    cfg = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration_ms": 30,
            "input_device": 3,
        }
    }
    rec, _ = _make_recorder(cfg)

    mock_stream = MagicMock()
    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        rec.start_recording()

    _, kwargs = mock_sd.InputStream.call_args
    assert kwargs["samplerate"] == 16000
    assert kwargs["channels"] == 1
    assert kwargs["blocksize"] == 480
    assert kwargs["device"] == 3
    assert kwargs["dtype"] == "float32"
    assert kwargs["callback"] == rec._audio_callback

    rec._active = False
