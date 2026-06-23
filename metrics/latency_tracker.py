import time
from typing import Dict, Any

class LatencyTracker:
    """
    Measures and tracks execution latencies across the different stages of the voice pipeline
    (Noise cancellation, VAD boundary check, STT, LLM TTFT, TTS Synthesis, Player Playback).
    Essential for identifying and resolving conversational delay bottlenecks.
    """

    def __init__(self):
        """
        Initializes trackers with dictionary metrics structures.
        """
        self._latencies: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start_stage(self, stage_name: str) -> None:
        """
        Records the start timestamp for a specified pipeline stage.

        Args:
            stage_name (str): Identifier of the stage (e.g. 'stt', 'llm_ttft').
        """
        self._start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """
        Calculates and records the elapsed duration for a specified pipeline stage.

        Args:
            stage_name (str): Identifier of the stage.

        Returns:
            float: The duration in seconds, or 0.0 if start_stage was not called first.
        """
        start = self._start_times.get(stage_name)
        if start is None:
            return 0.0

        elapsed = time.perf_counter() - start
        self._latencies[stage_name] = elapsed
        del self._start_times[stage_name]
        return elapsed

    def get_latencies(self) -> Dict[str, float]:
        """
        Retrieves all currently recorded stage latencies.

        Returns:
            Dict[str, float]: Stage identifiers mapped to float durations in seconds.
        """
        return self._latencies.copy()

    def clear(self) -> None:
        """
        Resets latency statistics state.
        """
        self._latencies.clear()
        self._start_times.clear()
