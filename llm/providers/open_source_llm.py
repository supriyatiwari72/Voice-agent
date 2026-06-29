import json
import logging
import time
import requests
from typing import Dict, Any, Generator
from llm.base import BaseLLM

logger = logging.getLogger(__name__)

class OpenSourceLLM(BaseLLM):
    """
    Concrete adapter for open-source and OpenAI-compatible Large Language Model (LLM) backends.
    Supports local endpoints (Ollama, vLLM, llama.cpp) and cloud APIs (Groq, OpenRouter).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OpenSourceLLM provider using configurations.
        """
        self.config = config or {}
        
        # Load parameters from models_meta configurations
        models_meta = self.config.get("models_meta", {})
        llm_config = models_meta.get("llm_providers", {}).get("qwen", {}) or self.config

        self.api_base = llm_config.get("api_base", "http://localhost:11434")
        self.api_key = llm_config.get("api_key", "dummy-key")
        self.model_name = llm_config.get("model_name", "qwen-2.5-72b")
        self.temperature = float(llm_config.get("temperature", 0.7))
        self.max_tokens = int(llm_config.get("max_tokens", 150))
        self.timeout = float(llm_config.get("timeout_seconds", 10.0))
        self.max_retries = int(llm_config.get("max_retries", 3))

        # Dynamically build the standard OpenAI chat completions endpoint URL
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
            "Authorization": f"Bearer {self.api_key}"
        }

        logger.info(
            f"Initialized OpenSourceLLM (Qwen): target_url={self.url}, "
            f"model_name={self.model_name}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, timeout_seconds={self.timeout}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a complete text response for a given text prompt.

        Args:
            prompt (str): User input text.

        Returns:
            str: Full response text.
        """
        if not prompt:
            return ""

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        # Attempt call with built-in retry mechanism
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending LLM batch completion request (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                if content:
                    print("LLM Response:")
                    print(content.strip())
                return content.strip()

            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"LLM batch generation failed completely after {self.max_retries} attempts.")
                    return "Error: Unable to connect to LLM."
                time.sleep(1.0)
        
        return "Error: Unable to connect to LLM."

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream response tokens from the LLM.

        Args:
            prompt (str): User input text.

        Yields:
            str: Response token string.
        """
        if not prompt:
            return

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        response = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Initiating LLM stream request (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM stream connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"LLM stream generation failed completely after {self.max_retries} attempts.")
                    yield "Error: LLM stream connection failed."
                    return
                time.sleep(1.0)

        # Parse SSE stream line-by-line
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk_json = json.loads(data_str)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError as je:
                        logger.warning(f"Failed to decode stream chunk JSON: {je} for data: {data_str}")
        except Exception as e:
            logger.error(f"Exception encountered while processing LLM SSE streams: {e}")
            yield " [Stream Error] "
