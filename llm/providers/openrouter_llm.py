import json
import logging
import os
import time
from typing import Dict, Any, Generator, Optional
from llm.base import BaseLLM

logger = logging.getLogger(__name__)

class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM Provider — single API key to access multiple models.
    Uses OpenAI-compatible API via openai Python client.

    Free models available: google/gemini-2.0-flash-exp:free, meta-llama/llama-4-maverick:free, etc.
    Set OPENROUTER_API_KEY in .env or config/models.yaml.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        llm_config = models_meta.get("llm_providers", {}).get("openrouter", {}) or {}

        self.api_key = (
            llm_config.get("api_key")
            or self.config.get("openrouter_api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        )
        self.api_base = llm_config.get("api_base", "https://openrouter.ai/api/v1").rstrip("/")
        self.model_name = llm_config.get("model_name", "google/gemini-2.0-flash-exp:free")
        self.temperature = float(llm_config.get("temperature", 0.7))
        self.max_tokens = int(llm_config.get("max_tokens", 200))
        self.timeout = float(llm_config.get("timeout_seconds", 30.0))
        self.max_retries = int(llm_config.get("max_retries", 3))
        self.system_prompt = self.config.get("system_prompt", "")

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        logger.info(
            f"OpenRouterLLM: model={self.model_name}, url={self.api_base}, "
            f"api_key={'set' if self.api_key else 'MISSING'}"
        )

        if not self.api_key:
            logger.warning(
                "OpenRouterLLM: No OPENROUTER_API_KEY found. Set OPENROUTER_API_KEY in .env or config. "
                "Falling back to mock streaming."
            )
            self.client = None

    def _build_messages(self, prompt: str):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _mock_stream(self, reason: str = "") -> Generator[str, None, None]:
        logger.warning(f"OpenRouterLLM: Mock fallback{', reason: ' + reason if reason else ''}.")
        mock = (
            "I'm running in mock mode. To enable real OpenRouter responses, "
            "set OPENROUTER_API_KEY in your .env file."
        )
        for word in mock.split():
            yield word + " "

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        if not prompt:
            return

        if not self.client or not self.api_key:
            yield from self._mock_stream(reason="No API key")
            return

        messages = self._build_messages(prompt)
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                content = delta.content
                if not content and getattr(delta, "reasoning", None):
                    content = delta.reasoning
                if content:
                    yield content
        except Exception as e:
            logger.error(f"OpenRouterLLM stream error: {e}")
            yield from self._mock_stream(reason=str(e))

    def generate(self, prompt: str) -> str:
        if not prompt:
            return ""
        return "".join(self.generate_stream(prompt))
