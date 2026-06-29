import json
import logging
import os
import random
import requests
import time
from typing import Dict, Any, Generator, Optional
from llm.base import BaseLLM

logger = logging.getLogger(__name__)

# Groq free tier rate limits (as of 2026):
#   llama-3.1-8b-instant:  30 req/min, 15k tpm  (128K context window)
#   llama3-8b-8192:         30 req/min, 15k tpm  (8K context window)
#   gemma2-9b-it:           30 req/min, 15k tpm  (8K context window)
#   mixtral-8x7b-32768:     30 req/min, 15k tpm  (32K context window)
_RATE_LIMIT_REQUESTS_PER_MINUTE = 30


class GroqLLM(BaseLLM):
    """
    Groq LLM Provider — free, ultra-low-latency inference via Groq's API.

    Free tier: https://console.groq.com  (no credit card required)
    Default model: llama-3.1-8b-instant (128K context, fastest free option)

    Key features:
      - System prompt injection from config
      - Exponential backoff with jitter for rate limits (HTTP 429)
      - Per-minute rate limit tracking (pauses when approaching limit)
      - SSE streaming with robust error recovery
      - Mock fallback when no API key is set

    Set GROQ_API_KEY in .env or config/models.yaml.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        models_meta = self.config.get("models_meta", {})
        groq_config = models_meta.get("llm_providers", {}).get("groq", {}) or {}

        # Resolve API key: config -> env var
        self._api_key = (
            groq_config.get("api_key")
            or self.config.get("groq_api_key")
            or os.environ.get("GROQ_API_KEY")
            or ""
        )

        self.api_base = groq_config.get("api_base", "https://api.groq.com/openai/v1").rstrip("/")
        self.model_name = groq_config.get("model_name", "llama-3.1-8b-instant")
        self.temperature = float(groq_config.get("temperature", 0.7))
        self.max_tokens = int(groq_config.get("max_tokens", 200))
        self.timeout = float(groq_config.get("timeout_seconds", 15.0))
        self.max_retries = int(groq_config.get("max_retries", 3))

        # System prompt from root config
        self.system_prompt = self.config.get("system_prompt", "")

        # Build URL
        base_url = self.api_base.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            if base_url.endswith("/v1"):
                self.url = f"{base_url}/chat/completions"
            else:
                self.url = f"{base_url}/v1/chat/completions"
        else:
            self.url = base_url

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Rate limit tracking
        self._req_timestamps = []

        logger.info(
            f"GroqLLM: model={self.model_name}, url={self.url}, "
            f"api_key={'set' if self._api_key else 'MISSING'}, "
            f"system_prompt={'set' if self.system_prompt else 'none'}"
        )

        self._groq_api_key = self._api_key  # backward compat alias for tests

        if not self._api_key:
            logger.warning(
                "GroqLLM: No GROQ_API_KEY found. Set GROQ_API_KEY in .env or config. "
                "Falling back to mock streaming."
            )

    # ------------------------------------------------------------------
    # Rate limiter
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        """Enforce per-minute rate limit by sleeping if necessary."""
        now = time.time()
        window = 60.0
        # Remove timestamps older than 1 minute
        self._req_timestamps = [t for t in self._req_timestamps if now - t < window]

        if len(self._req_timestamps) >= _RATE_LIMIT_REQUESTS_PER_MINUTE:
            oldest = min(self._req_timestamps)
            sleep_time = window - (now - oldest) + 0.5
            if sleep_time > 0:
                logger.info(f"GroqLLM: Rate limit approaching. Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self._req_timestamps.append(time.time())

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str) -> list:
        """Build messages array with optional system prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Mock fallback
    # ------------------------------------------------------------------

    def _mock_stream(self, reason: str = "") -> Generator[str, None, None]:
        """Yield a mock response when no API key is configured."""
        logger.warning(f"GroqLLM: Mock fallback{', reason: ' + reason if reason else ''}.")
        mock = (
            "I'm running in mock mode. To enable real Groq responses, "
            "set GROQ_API_KEY in your .env file."
        )
        for word in mock.split():
            yield word + " "

    # ------------------------------------------------------------------
    # HTTP request with retry + rate limiting
    # ------------------------------------------------------------------

    def _post(self, payload: dict, stream: bool = False) -> Optional[requests.Response]:
        """
        Send POST request with exponential backoff, rate-limit handling,
        and jitter. Returns response on success, None on total failure.
        """
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()

                resp = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    stream=stream,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    return resp

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    wait = retry_after + random.uniform(0, 2)
                    logger.warning(
                        f"GroqLLM: Rate limited (429). Retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code in (401, 403):
                    logger.error(f"GroqLLM: Authentication failed ({resp.status_code}). Check API key.")
                    return None

                if resp.status_code >= 500:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"GroqLLM: Server error ({resp.status_code}). "
                        f"Retrying in {wait:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"GroqLLM: Unexpected HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            except requests.exceptions.Timeout:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"GroqLLM: Timeout (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

            except requests.exceptions.ConnectionError as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"GroqLLM: Connection error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

            except requests.exceptions.RequestException as e:
                logger.error(f"GroqLLM: Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(1.0)

        return None

    # ------------------------------------------------------------------
    # Stream generation
    # ------------------------------------------------------------------

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream tokens from Groq's API with robust error handling."""
        if not prompt:
            return

        if not self._api_key:
            yield from self._mock_stream(reason="No API key")
            return

        messages = self._build_messages(prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        response = self._post(payload, stream=True)
        if response is None:
            yield from self._mock_stream(reason="API request failed after retries")
            return

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"GroqLLM: Stream connection broken mid-response: {e}")
        except Exception as e:
            logger.error(f"GroqLLM: Stream error: {e}")

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Batch generation — joins the stream into a single string."""
        if not prompt:
            return ""
        return "".join(self.generate_stream(prompt))
