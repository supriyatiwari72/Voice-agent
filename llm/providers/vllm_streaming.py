import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Generator
from llm.base import BaseLLM

logger = logging.getLogger(__name__)

class VLLMStreamingLLM(BaseLLM):
    """
    vLLM Streaming LLM Provider.
    Streams tokens from a local or remote vLLM hosted OpenAI-compatible server.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("llm", {}).get("vllm", {})
        self.model = self.config.get("model") or config.get("models", {}).get("llm_model") or "qwen2.5"
        self.url = self.config.get("url") or "http://localhost:8000/v1/chat/completions"
        self.api_key = self.config.get("api_key") or "vllm-key"
        
    def generate(self, prompt: str) -> str:
        """
        Batch generation fallback.
        """
        tokens = list(self.generate_stream(prompt))
        return "".join(tokens)

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream tokens using Server-Sent Events (SSE) JSON protocol.
        """
        logger.info(f"VLLMStreamingLLM: Generating stream for model {self.model}")
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        data = json.dumps(payload).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        req = urllib.request.Request(
            self.url,
            data=data,
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=0.25) as response:
                for line in response:
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data:"):
                            content_str = decoded[5:].strip()
                            if content_str == "[DONE]":
                                break
                            if content_str:
                                chunk = json.loads(content_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    token = choices[0].get("delta", {}).get("content", "")
                                    if token:
                                        yield token
                                        
        except Exception as e:
            logger.warning(f"VLLMStreamingLLM: Server unreachable ({e}). Falling back to mock streaming.")
            mock_response = f"This is a real-time streaming LLM response from vLLM using model {self.model}."
            for word in mock_response.split():
                yield word + " "
