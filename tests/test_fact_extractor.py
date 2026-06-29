import pytest
from unittest.mock import MagicMock
from memory.fact_extractor import FactExtractor

def test_fact_extractor_success():
    """
    Verify regex-based extraction finds the user's name correctly.
    """
    mock_llm = MagicMock()
    extractor = FactExtractor(mock_llm)
    facts = extractor.extract_facts("My name is Supriya.")

    assert facts == {"name": "Supriya"}
    # Regex extractor does not call LLM
    mock_llm.generate.assert_not_called()

def test_fact_extractor_markdown_json():
    """
    Verify parsing regex works when response contains enclosing markdown tags.
    """
    mock_llm = MagicMock()
    mock_llm.generate.return_value = 'Here are the facts: ```json {"name": "Supriya"} ```'
    
    extractor = FactExtractor(mock_llm)
    facts = extractor.extract_facts("Hello, my name is Supriya.")
    
    assert facts == {"name": "Supriya"}

def test_fact_extractor_empty_or_short_input():
    """
    Verify short inputs return empty dictionary without LLM execution.
    """
    mock_llm = MagicMock()
    extractor = FactExtractor(mock_llm)
    
    assert extractor.extract_facts("") == {}
    assert extractor.extract_facts("hi") == {}
    mock_llm.generate.assert_not_called()

def test_fact_extractor_invalid_json():
    """
    Verify parser handles invalid JSON responses safely, returning empty dictionary.
    """
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "No facts found."
    
    extractor = FactExtractor(mock_llm)
    facts = extractor.extract_facts("Some random query here.")
    assert facts == {}
