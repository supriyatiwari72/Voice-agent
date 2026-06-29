import json
import logging
import os
import urllib.request
import urllib.error
import time
from typing import Dict, Any, Generator, Optional
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaStreamingLLM(BaseLLM):
    """
    Ollama Streaming LLM Provider.
    Streams tokens from a local Ollama server using the /api/chat endpoint.
    Supports system prompt injection and graceful fallback when Ollama is unreachable.
    """

    def __init__(self, config: Dict[str, Any]):
        # Config resolution: models_meta → llm_providers → ollama (or per-model key)
        models_meta = config.get("models_meta", {})
        llm_providers = models_meta.get("llm_providers", {})

        # Support both the generic "ollama" key and specific model keys (qwen2.5_3b etc.)
        # The factory closures for specific models inject their config under the "ollama" key.
        llm_config = llm_providers.get("ollama", {}) or {}

        self.model = (
            llm_config.get("model_name")
            or config.get("models", {}).get("llm_model")
            or "qwen2.5:3b"
        )

        api_base = (
            llm_config.get("api_base")
            or config.get("ollama_api_base")
            or "http://localhost:11434"
        ).rstrip("/")

        # Use /api/chat (supports messages array with system role)
        self.url = f"{api_base}/api/chat"
        self.temperature = float(llm_config.get("temperature", 0.7))
        self.max_tokens = int(llm_config.get("max_tokens", 200))
        self.timeout = float(llm_config.get("timeout_seconds", 30.0))
        self.max_retries = int(llm_config.get("max_retries", 2))

        # System prompt from config (global) or model-specific config
        self.system_prompt: Optional[str] = (
            config.get("system_prompt")
            or llm_config.get("system_prompt")
            or os.environ.get("VOICE_AGENT_SYSTEM_PROMPT")
        )

        logger.info(
            f"OllamaStreamingLLM: model={self.model}, url={self.url}, "
            f"timeout={self.timeout}s, system_prompt={'set' if self.system_prompt else 'none'}"
        )

        # Health check and warmup on startup
        val_start = time.time()
        healthy = self.health_check()
        warmed_up = False
        if healthy:
            warmed_up = self.warmup()
        
        latency = time.time() - val_start
        status_str = "Healthy" if healthy else "Unhealthy"
        logger.info(f"Ollama Startup Log: Model={self.model} | Startup Latency={latency:.3f}s | Endpoint Status={status_str} | Warmup Success={warmed_up}")
        if healthy and warmed_up:
            print("LLM Ready")

    def health_check(self) -> bool:
        """
        Verify Ollama is reachable, the endpoint is healthy, and the model is available.
        """
        try:
            api_base = self.url.rsplit("/api/chat", 1)[0]
            show_url = f"{api_base}/api/show"
            payload = {"name": self.model}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                show_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            # Use a short timeout of 5 seconds for health checks
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    if "modelfile" in resp_data or "details" in resp_data:
                        return True
        except Exception as e:
            logger.warning(f"Ollama health check failed for {self.model}: {e}")
        return False

    def warmup(self) -> bool:
        """
        Warm up the model by sending a small prompt.
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {
                    "num_predict": 1,
                },
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            # Use a slightly larger timeout for warmup if the model needs loading
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    json.loads(response.read().decode("utf-8"))
                    return True
        except Exception as e:
            logger.warning(f"Ollama warmup failed for {self.model}: {e}")
        return False

    def _build_messages(self, prompt: str) -> list:
        """Build the messages array for the chat API."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _mock_stream(self, reason: str = "") -> Generator[str, None, None]:
        """Yield a mock response when Ollama is unreachable."""
        logger.warning(
            f"OllamaStreamingLLM: Falling back to mock stream "
            f"(model={self.model}{', reason: ' + reason if reason else ''})."
        )
        mock = (
            f"I'm running in offline mode. The Ollama server is not reachable. "
            f"To enable real responses, start Ollama and run: ollama pull {self.model}"
        )
        for word in mock.split():
            yield word + " "

    def generate(self, prompt: str) -> str:
        """Batch generation — joins the stream into a single string."""
        return "".join(self.generate_stream(prompt))

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream tokens from the local Ollama /api/chat endpoint.
        Gracefully falls back to mock tokens if Ollama is not running.
        """
        if not prompt:
            return

        # Pre-request health check verification
        if not self.health_check():
            yield from self._mock_stream(reason="Pre-request health check failed")
            return

        messages = self._build_messages(prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8").strip())
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

        except urllib.error.URLError as e:
            yield from self._mock_stream(reason=str(e))
        except Exception as e:
            logger.error(f"OllamaStreamingLLM: Unexpected error: {e}")
            yield from self._mock_stream(reason=str(e))
