import pytest
from unittest.mock import patch, MagicMock
import requests
from llm.factory import LLMFactory
from llm.providers.open_source_llm import OpenSourceLLM
from llm.base import BaseLLM

@pytest.fixture
def mock_llm_config():
    return {
        "models_meta": {
            "llm_providers": {
                "qwen": {
                    "api_base": "https://api.groq.com/openai/v1",
                    "api_key": "test-key-123",
                    "model_name": "qwen-2.5-72b",
                    "temperature": 0.5,
                    "max_tokens": 80,
                    "timeout_seconds": 2.0,
                    "max_retries": 2
                }
            }
        }
    }

def test_open_source_llm_factory_creation(mock_llm_config):
    """
    Verify that LLMFactory correctly resolves 'qwen' and 'ollama' to OpenSourceLLM.
    """
    provider_qwen = LLMFactory.get_provider("qwen", mock_llm_config)
    assert isinstance(provider_qwen, OpenSourceLLM)
    assert isinstance(provider_qwen, BaseLLM)

    provider_ollama = LLMFactory.get_provider("ollama", mock_llm_config)
    assert isinstance(provider_ollama, OpenSourceLLM)

def test_open_source_llm_url_normalization(mock_llm_config):
    """
    Verify that URLs are built correctly regardless of the trailing paths.
    """
    config_1 = {
        "models_meta": {"llm_providers": {"qwen": {"api_base": "http://localhost:11434"}}}
    }
    llm_1 = OpenSourceLLM(config_1)
    assert llm_1.url == "http://localhost:11434/v1/chat/completions"

    config_2 = {
        "models_meta": {"llm_providers": {"qwen": {"api_base": "https://api.groq.com/openai/v1"}}}
    }
    llm_2 = OpenSourceLLM(config_2)
    assert llm_2.url == "https://api.groq.com/openai/v1/chat/completions"

    config_3 = {
        "models_meta": {"llm_providers": {"qwen": {"api_base": "http://localhost:8000/v1/chat/completions"}}}
    }
    llm_3 = OpenSourceLLM(config_3)
    assert llm_3.url == "http://localhost:8000/v1/chat/completions"

@patch("requests.post")
def test_open_source_llm_generate_success(mock_post, mock_llm_config):
    """
    Verify that generate() parses standard OpenAI chat JSON response correctly.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hi there! I am Qwen."
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    llm = OpenSourceLLM(mock_llm_config)
    result = llm.generate("Hello")
    assert result == "Hi there! I am Qwen."
    mock_post.assert_called_once()

@patch("requests.post")
def test_open_source_llm_generate_retry_and_fail(mock_post, mock_llm_config):
    """
    Verify that generate() retries the specified number of times on connection failures.
    """
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

    llm = OpenSourceLLM(mock_llm_config)
    # With max_retries=2, it will retry twice
    result = llm.generate("Hello")
    
    assert "Error:" in result
    assert mock_post.call_count == 2

@patch("requests.post")
def test_open_source_llm_generate_stream(mock_post, mock_llm_config):
    """
    Verify that generate_stream() parses SSE stream data chunks successfully.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
        b"", # Empty line should be skipped
        b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}",
        b"data: [DONE]"
    ]
    mock_post.return_value = mock_response

    llm = OpenSourceLLM(mock_llm_config)
    stream = llm.generate_stream("Hello")
    chunks = list(stream)
    
    assert chunks == ["Hello", " world"]
    mock_post.assert_called_once()

@patch("requests.post")
def test_open_source_llm_malformed_response(mock_post, mock_llm_config):
    """
    Verify that generate() handles malformed JSON responses and missing fields gracefully.
    """
    # 1. JSON decoding error
    mock_response_err = MagicMock()
    mock_response_err.status_code = 200
    mock_response_err.json.side_effect = ValueError("Invalid JSON syntax")
    mock_post.return_value = mock_response_err
    
    llm = OpenSourceLLM(mock_llm_config)
    result = llm.generate("Hello")
    assert "Error:" in result

    # 2. JSON decoding of stream with invalid format
    mock_post.reset_mock()
    mock_response_stream = MagicMock()
    mock_response_stream.status_code = 200
    mock_response_stream.iter_lines.return_value = [
        b"data: {invalid json here}",
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Valid\"}}]}",
        b"data: [DONE]"
    ]
    mock_post.return_value = mock_response_stream
    
    stream = llm.generate_stream("Hello")
    chunks = list(stream)
    assert chunks == ["Valid"]
