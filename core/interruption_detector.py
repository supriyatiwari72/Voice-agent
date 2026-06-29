import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that indicate a user wants to interrupt/pause the assistant
_INTERRUPT_PATTERNS = [
    re.compile(r"\b(wait|hold on|hold up|hang on|one sec|one moment)\b", re.IGNORECASE),
    re.compile(r"\b(stop|pause|freeze|enough|that.s enough|quiet|silence)\b", re.IGNORECASE),
    re.compile(r"\b(interrupt|barge.in|cut in|let me speak|let me talk)\b", re.IGNORECASE),
    re.compile(r"\b(actually|but|however|wait wait|hold it)\b", re.IGNORECASE),
]

_GOODBYE_PATTERNS = [
    re.compile(r"\b(goodbye|bye bye|see you|talk to you later|that.s all|we.re done)\b", re.IGNORECASE),
    re.compile(r"\b(stop the conversation|end the call|end call|quit|exit)\b", re.IGNORECASE),
]


def is_interruption(text: str) -> bool:
    """
    Returns True if the transcribed text contains an interruption keyword
    that should pause the assistant's current response.
    """
    for pattern in _INTERRUPT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def is_goodbye(text: str) -> bool:
    """
    Returns True if the transcribed text contains a farewell phrase
    that should end the conversation.
    """
    for pattern in _GOODBYE_PATTERNS:
        if pattern.search(text):
            return True
    return False


class InterruptionDetector:
    """
    Monitors partial/final transcripts for interruption keywords during
    assistant speech and returns structured interruption signals.
    """

    @staticmethod
    def detect_signal(text: str) -> Optional[str]:
        """Returns 'interrupt', 'goodbye', or None."""
        if not text:
            return None
        if is_goodbye(text):
            return "goodbye"
        if is_interruption(text):
            return "interrupt"
        return None
