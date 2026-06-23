import logging
from noise.base import BaseNoiseCanceller
from vad.base import BaseVAD
from stt.base import BaseSTT
from llm.base import BaseLLM
from tts.base import BaseTTS
from pipeline.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class VoicePipeline:
    """
    Main orchestration loop that coordinates passing audio and text frames between:
    BaseNoiseCanceller -> BaseVAD -> BaseSTT -> BaseLLM -> BaseTTS.

    This class adheres to the Dependency Inversion Principle; it depends only on abstract
    interfaces, never on concrete providers (e.g., Whisper, Gemini, ElevenLabs).
    """

    def __init__(
        self,
        noise_canceller: BaseNoiseCanceller,
        vad: BaseVAD,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS
    ):
        self.noise_canceller = noise_canceller
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self._state = PipelineState.IDLE
        self._speech_buffer = b""
        self._in_speech = False

    def set_state(self, state: PipelineState) -> None:
        self._state = state

    def get_state(self) -> PipelineState:
        return self._state

    def process_frame(self, raw_frame: bytes) -> bytes:
        """
        Processes a single incoming audio frame through the sequential MVP V1 pipeline:
        1. Noise cancellation filters background artifacts.
        2. VAD decides if the frame contains speech.
        3. Transcribe audio to text if speech boundary is completed.
        4. Generate response via LLM.
        5. Synthesize text to speech.

        Args:
            raw_frame (bytes): Incoming audio bytes from microphone.

        Returns:
            bytes: The resulting synthesized audio output bytes or empty bytes if silent.
        """
        # 1. Noise Cancellation
        cleaned_frame = self.noise_canceller.process(raw_frame)

        # 2. VAD Speech Gating Check
        is_speech = self.vad.is_speech(cleaned_frame)

        if is_speech:
            self.set_state(PipelineState.LISTENING)
            self._in_speech = True
            self._speech_buffer += cleaned_frame
            return b""
        
        # If transitioning from speech to silence: process the complete utterance
        if not is_speech and self._in_speech:
            self.set_state(PipelineState.PROCESSING)
            
            # 3. Speech To Text
            self.set_state(PipelineState.TRANSCRIBING)
            text = self.stt.transcribe(self._speech_buffer)
            
            # 4. LLM Generation
            self.set_state(PipelineState.THINKING)
            response_text = self.llm.generate(text)
            
            # 5. Text To Speech
            self.set_state(PipelineState.SPEAKING)
            synthesized_audio = self.tts.synthesize(response_text)
            
            # Clean up turn states
            self._speech_buffer = b""
            self._in_speech = False
            self.set_state(PipelineState.IDLE)
            
            return synthesized_audio

        # No speech active
        return b""
