import logging
from typing import Any

logger = logging.getLogger(__name__)

class MemoryMetrics:
    """
    Telemetry recorder mapping memory pipeline metrics to the central MetricsTracker.
    """
    def __init__(self, metrics_tracker: Any):
        self.metrics_tracker = metrics_tracker

    def record_turn_count(self, count: int) -> None:
        if self.metrics_tracker:
            self.metrics_tracker.record_metric("memory_turn_count", float(count))

    def record_summary_count(self, count: int) -> None:
        if self.metrics_tracker:
            self.metrics_tracker.record_metric("summary_count", float(count))

    def record_summary_generation_time(self, time_ms: float) -> None:
        if self.metrics_tracker:
            self.metrics_tracker.record_metric("summary_generation_time_ms", time_ms)

    def record_context_build_time(self, time_ms: float) -> None:
        if self.metrics_tracker:
            self.metrics_tracker.record_metric("context_build_time_ms", time_ms)

    def record_context_size(self, size_chars: int) -> None:
        if self.metrics_tracker:
            self.metrics_tracker.record_metric("average_context_size", float(size_chars))
