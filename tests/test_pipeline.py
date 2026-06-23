from noise.factory import NoiseFactory
from vad.factory import VADFactory
from stt.factory import STTFactory
from llm.factory import LLMFactory
from tts.factory import TTSFactory
from pipeline.voice_pipeline import VoicePipeline
from pipeline.pipeline_state import PipelineState

def test_pipeline_initialization_and_states():
    """
    Test that VoicePipeline correctly constructs from abstract references and switches state.
    """
    # 1. Resolve providers from registries
    noise = NoiseFactory.get_provider("rnnoise")
    vad = VADFactory.get_provider("dummy")
    stt = STTFactory.get_provider("faster_whisper")
    llm = LLMFactory.get_provider("gemini")
    tts = TTSFactory.get_provider("elevenlabs")

    # 2. Build Pipeline
    pipeline = VoicePipeline(
        noise_canceller=noise,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts
    )

    # 3. Verify setup state
    assert pipeline.get_state() == PipelineState.IDLE

    # 4. Toggle states and verify behavior
    pipeline.set_state(PipelineState.LISTENING)
    assert pipeline.get_state() == PipelineState.LISTENING

    pipeline.set_state(PipelineState.SPEAKING)
    assert pipeline.get_state() == PipelineState.SPEAKING

def test_pipeline_process_frame_compilation():
    """
    Test that process_frame executes cleanly.
    """
    noise = NoiseFactory.get_provider("rnnoise")
    vad = VADFactory.get_provider("dummy")
    stt = STTFactory.get_provider("faster_whisper")
    llm = LLMFactory.get_provider("gemini")
    tts = TTSFactory.get_provider("elevenlabs")

    pipeline = VoicePipeline(
        noise_canceller=noise,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts
    )

    # First frame: VAD detects speech, returns empty bytes (accumulating buffer)
    dummy_in = b"\x00\x01\x02"
    dummy_out_1 = pipeline.process_frame(dummy_in)
    assert dummy_out_1 == b""
    assert pipeline._in_speech is True

    # Second frame: VAD returns false (silence), triggers STT -> LLM -> TTS pipeline
    dummy_out_2 = pipeline.process_frame(dummy_in)
    
    # In Phase 1 dummy providers, the TTS synthesizes 100 bytes of b"\x00"
    assert dummy_out_2 == b"\x00" * 100
    assert pipeline._in_speech is False

