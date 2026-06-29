import pytest
from unittest.mock import MagicMock
from memory.models import Turn
from memory.summarizer import MemorySummarizer

def test_summarizer_success():
    """
    Verify summarizer builds prompt and invokes llm.generate() successfully.
    """
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Merged summary of AI turns."
    
    summarizer = MemorySummarizer(mock_llm)
    
    turns = [
        Turn("hello", "hi"),
        Turn("what is AI?", "artificial intelligence")
    ]
    
    new_summary = summarizer.generate_summary("Prior context summary.", turns)
    
    assert new_summary == "Merged summary of AI turns."
    mock_llm.generate.assert_called_once()
    
    # Verify turns were serialized inside prompt
    call_args = mock_llm.generate.call_args[0][0]
    assert "User: hello" in call_args
    assert "Assistant: hi" in call_args
    assert "what is AI?" in call_args

def test_summarizer_empty_turns():
    """
    Verify empty turns list returns existing summary without calling LLM.
    """
    mock_llm = MagicMock()
    summarizer = MemorySummarizer(mock_llm)
    
    new_summary = summarizer.generate_summary("Existing summary.", [])
    assert new_summary == "Existing summary."
    mock_llm.generate.assert_not_called()

def test_summarizer_llm_failure():
    """
    Verify summarizer falls back gracefully if LLM raises exception.
    """
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("Ollama Timeout")
    
    summarizer = MemorySummarizer(mock_llm)
    
    turns = [Turn("test user", "test assistant")]
    
    # Should not raise exception, but return fallback text
    new_summary = summarizer.generate_summary("Old summary.", turns)
    assert "Old summary." in new_summary
    assert "test user" in new_summary
