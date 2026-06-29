import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Regex patterns for common user facts — ordered by priority
_FACT_PATTERNS = [
    # Name patterns — stop at conjunctions, punctuation, or sentence end
    ("name", re.compile(
        r"(?:my name is|i am|i'm|call me|they call me|people call me|you can call me)\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?)(?:\s+(?:and|but|or|so)\b|[,\.!?]|$)",
        re.IGNORECASE
    )),
    # Age patterns
    ("age", re.compile(
        r"i(?:'m| am)\s+(\d{1,3})\s+years?\s+old|my age is\s+(\d{1,3})",
        re.IGNORECASE
    )),
    # Location patterns
    ("location", re.compile(
        r"(?:i live in|i'm from|i am from|i'm in|i stay in)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$)",
        re.IGNORECASE
    )),
    # Occupation patterns
    ("occupation", re.compile(
        r"(?:i(?:'m| am) (?:a|an)\s+)([a-zA-Z\s]+?)(?:\s+(?:by profession|by trade|at |\.|,|$))",
        re.IGNORECASE
    )),
]


class FactExtractor:
    """
    Extracts key-value facts from user messages using fast regex patterns.
    Catches name introductions, age, location and occupation instantly
    without any LLM API call, adding zero latency to the voice pipeline.
    """
    def __init__(self, llm: Any):
        # llm kept for API compatibility but not used in this implementation
        self.llm = llm

    def extract_facts(self, user_message: str) -> Dict[str, Any]:
        """
        Scans user message with regex patterns and returns any found facts.
        """
        if not user_message or len(user_message.strip()) < 3:
            return {}

        facts: Dict[str, Any] = {}

        for fact_key, pattern in _FACT_PATTERNS:
            match = pattern.search(user_message)
            if match:
                # Pick the first non-None group
                value = next((g for g in match.groups() if g is not None), None)
                if value:
                    value = value.strip().rstrip(".,!?")
                    facts[fact_key] = value
                    logger.info(f"FactExtractor: Extracted '{fact_key}' = '{value}'")

        return facts
