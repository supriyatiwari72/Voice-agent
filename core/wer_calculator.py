import logging

logger = logging.getLogger(__name__)


def _words(text: str) -> list[str]:
    """Split text into normalized word list."""
    return text.strip().lower().split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis strings
    using Levenshtein (edit) distance at the word level.

    Returns a float in [0.0, inf) — lower is better. 0.0 means perfect.
    """
    ref_words = _words(reference)
    hyp_words = _words(hypothesis)

    ref_len = len(ref_words)
    hyp_len = len(hyp_words)

    # dp[i][j] = edit distance between ref_words[:i] and hyp_words[:j]
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[ref_len][hyp_len] / max(ref_len, 1)


class WERTracker:
    """
    Tracks Word Error Rate across multiple turns for STT accuracy evaluation.
    """

    def __init__(self):
        self._scores: list[float] = []

    def record(self, reference: str, hypothesis: str) -> float:
        wer = word_error_rate(reference, hypothesis)
        self._scores.append(wer)
        return wer

    @property
    def average(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    @property
    def count(self) -> int:
        return len(self._scores)

    @property
    def last(self) -> float:
        return self._scores[-1] if self._scores else 0.0
