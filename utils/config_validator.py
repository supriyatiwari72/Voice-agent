from typing import Dict, Any
from noise.factory import NoiseFactory
from vad.factory import VADFactory
from stt.factory import STTFactory
from llm.factory import LLMFactory
from tts.factory import TTSFactory

class ConfigValidationError(Exception):
    """
    Custom exception raised when configuration validation checks fail.
    """
    pass

class ConfigValidator:
    """
    Validates Voice-to-Voice AI Agent configuration keys and parameters
    prior to application startup to prevent runtime model errors.
    """

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """
        Validates the configuration structure and active providers.

        Args:
            config (Dict[str, Any]): Parsed application settings dictionary.

        Raises:
            ConfigValidationError: If validation checks fail.
        """
        if not config:
            raise ConfigValidationError("Configuration dictionary is empty.")

        # 1. Validate required parent sections
        required_sections = ["active_providers", "audio", "logging"]
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Missing required configuration section: '{section}'")

        # 2. Validate active providers values
        providers = config.get("active_providers", {})
        required_steps = ["noise", "vad", "stt", "llm", "tts"]
        for step in required_steps:
            if step not in providers:
                raise ConfigValidationError(f"Missing active provider configuration for pipeline step: '{step}'")

        # 3. Dynamic verification against registry catalogs
        noise_prov = providers["noise"]
        if noise_prov.strip().lower() not in NoiseFactory._providers:
            raise ConfigValidationError(
                f"Unsupported active noise provider '{noise_prov}'. "
                f"Supported: {list(NoiseFactory._providers.keys())}"
            )

        vad_prov = providers["vad"]
        if vad_prov.strip().lower() not in VADFactory._providers:
            raise ConfigValidationError(
                f"Unsupported active VAD provider '{vad_prov}'. "
                f"Supported: {list(VADFactory._providers.keys())}"
            )

        stt_prov = providers["stt"]
        if stt_prov.strip().lower() not in STTFactory._providers:
            raise ConfigValidationError(
                f"Unsupported active STT provider '{stt_prov}'. "
                f"Supported: {list(STTFactory._providers.keys())}"
            )

        llm_prov = providers["llm"]
        if llm_prov.strip().lower() not in LLMFactory._providers:
            raise ConfigValidationError(
                f"Unsupported active LLM provider '{llm_prov}'. "
                f"Supported: {list(LLMFactory._providers.keys())}"
            )

        tts_prov = providers["tts"]
        if tts_prov.strip().lower() not in TTSFactory._providers:
            raise ConfigValidationError(
                f"Unsupported active TTS provider '{tts_prov}'. "
                f"Supported: {list(TTSFactory._providers.keys())}"
            )

        # 4. Audio settings sanity check
        audio = config.get("audio", {})
        required_audio_keys = ["sample_rate", "channels", "bit_depth", "chunk_size"]
        for key in required_audio_keys:
            if key not in audio:
                raise ConfigValidationError(f"Missing required audio parameter: '{key}'")
            if not isinstance(audio[key], int) or audio[key] <= 0:
                raise ConfigValidationError(f"Audio parameter '{key}' must be a positive integer.")
