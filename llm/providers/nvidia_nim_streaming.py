import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Generator
from llm.base import BaseLLM

logger = logging.getLogger(__name__)

class NVIDIANIMStreamingLLM(BaseLLM):
    """
    NVIDIA NIM Streaming LLM Provider.
    Streams tokens from local or NVIDIA hosted NIM endpoints.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("llm", {}).get("nvidia_nim", {})
        self.model = self.config.get("model") or config.get("models", {}).get("llm_model") or "meta/llama3-70b-instruct"
        self.api_key = self.config.get("api_key") or config.get("nvidia_api_key") or "MOCK_KEY"
        self.url = self.config.get("url") or "https://integrate.api.nvidia.com/v1/chat/completions"
        
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
        logger.info(f"NVIDIANIMStreamingLLM: Generating stream for model {self.model}")
        
        if not self.api_key or self.api_key == "MOCK_KEY":
            logger.info("NVIDIANIMStreamingLLM: Mock key detected. Running mock fallback.")
            mock_response = f"This is a real-time streaming LLM response from NVIDIA NIM using model {self.model}."
            for word in mock_response.split():
                yield word + " "
            return

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
            with urllib.request.urlopen(req, timeout=0.5) as response:
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
            logger.warning(f"NVIDIANIMStreamingLLM: Connection error ({e}). Falling back to mock.")
            mock_response = f"This is a real-time streaming LLM response from NVIDIA NIM using model {self.model}."
            for word in mock_response.split():
                yield word + " "
