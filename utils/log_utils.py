import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class RateLimitedLogger:
    """
    Logs a warning only the first N times, then counts silently.
    Reports the total count when `report()` is called.

    Usage:
        rl = RateLimitedLogger("my_module", "Something unavailable", max_repeats=3)
        ...
        rl.warning()          # logs first 3 times, then just counts
        rl.report()           # "Something unavailable: triggered 47 times"
    """

    def __init__(self, name: str, message: str, max_repeats: int = 3):
        self._name = name
        self._message = message
        self._max_repeats = max_repeats
        self._count = 0
        self._first_time: float = 0.0

    def warning(self) -> None:
        if self._count == 0:
            self._first_time = time.time()
        self._count += 1
        if self._count <= self._max_repeats:
            logger.warning(f"{self._name}: {self._message}")
            if self._count == 1:
                print(f"[WARNING] {self._name}: {self._message}")

    def error(self, detail: str = "") -> None:
        self._count += 1
        if self._count <= self._max_repeats:
            logger.error(f"{self._name}: {self._message} {detail}".strip())

    def report(self) -> str:
        if self._count > self._max_repeats:
            extra = self._count - self._max_repeats
            msg = f"{self._name}: {self._message} — repeated {extra} more times (total {self._count})"
            logger.warning(msg)
            return msg
        return ""


# Global aggregator for per-module rate-limited loggers
_aggregators: Dict[str, RateLimitedLogger] = {}


def get_rate_limited(name: str, message: str, max_repeats: int = 3) -> RateLimitedLogger:
    key = f"{name}:{message}"
    if key not in _aggregators:
        _aggregators[key] = RateLimitedLogger(name, message, max_repeats)
    return _aggregators[key]


def report_all() -> None:
    """Call at shutdown to flush aggregated counts."""
    for agg in _aggregators.values():
        agg.report()
