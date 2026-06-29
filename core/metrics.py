import json
import logging
import os
import threading
from typing import Dict, Any, List, Optional
from core.wer_calculator import WERTracker

logger = logging.getLogger(__name__)

# ── Pretty latency formatting ────────────────────────────────────────────────

def _fmt(ms: float) -> str:
    if ms <= 0:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


class MetricsTracker:
    """
    Thread-safe registry coordinating runtime metrics collection and JSON exports.

    Extended with WER tracking and a formatted latency summary.
    """

    def __init__(self):
        """
        Initializes the in-memory metric lists.
        """
        self._lock = threading.Lock()
        self.metrics: Dict[str, List[float]] = {
            "vad_latency_ms": [],
            "stt_latency_ms": [],
            "llm_latency_ms": [],
            "tts_latency_ms": [],
            "ttft_ms": [],
            "total_turnaround_ms": [],
            "first_partial_transcript_ms": [],
            "first_llm_token_ms": [],
            "first_sentence_ms": [],
            "first_audio_chunk_ms": [],
            "speech_end_ms": [],
            "memory_turn_count": [],
            "summary_count": [],
            "summary_generation_time_ms": [],
            "context_build_time_ms": [],
            "average_context_size": [],
            "wer_score": [],
            "speech_detection_latency_ms": [],
            "eos_detection_latency_ms": [],
            "playback_latency_ms": [],
        }
        self.wer_tracker = WERTracker()

    def record_metric(self, name: str, value: float) -> None:
        """
        Safely records a floating-point latency value (in milliseconds).
        """
        with self._lock:
            if name in self.metrics:
                self.metrics[name].append(value)
                logger.debug(f"Recorded metric '{name}': {value:.2f} ms")
                
                # Performance target validations
                targets = {
                    "speech_detection_latency_ms": 100.0,
                    "eos_detection_latency_ms": 1000.0,
                    "first_partial_transcript_ms": 500.0,
                    "first_llm_token_ms": 700.0,
                    "ttft_ms": 700.0,
                    "first_audio_chunk_ms": 1500.0,
                    "total_turnaround_ms": 3000.0,
                    "playback_latency_ms": 100.0,
                }
                if name in targets and value > targets[name]:
                    logger.warning(
                        f"[Latency Target Warning] Metric '{name}' exceeded target: "
                        f"{value:.1f}ms > {targets[name]:.1f}ms"
                    )
            else:
                logger.warning(f"Attempted to record unregistered metric key: '{name}'")

    def record_wer(self, reference: str, hypothesis: str) -> float:
        """Record a WER score between a reference and STT transcript."""
        wer = self.wer_tracker.record(reference, hypothesis)
        self.record_metric("wer_score", wer)
        return wer

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Computes averages, counts, min and max limits for each recorded category.
        """
        summary = {}
        with self._lock:
            for name, values in self.metrics.items():
                if values:
                    summary[name] = {
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": float(len(values))
                    }
                else:
                    summary[name] = {
                        "average": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "count": 0.0
                    }
        return summary

    def print_latency_report(self) -> None:
        """Print a clean performance latency summary to the console."""
        s = self.get_summary()

        def _get(key: str) -> float:
            return s.get(key, {}).get("average", 0.0)

        def _fmt_ms(val: float) -> str:
            if val <= 0:
                return "N/A"
            return f"{val:.0f} ms"

        def _fmt_s(val: float) -> str:
            if val <= 0:
                return "N/A"
            return f"{val / 1000:.2f} s"

        stt = _get("stt_latency_ms")
        llm = _get("llm_latency_ms")
        tts = _get("tts_latency_ms")
        playback = _get("first_audio_chunk_ms")
        end_to_end = _get("total_turnaround_ms")

        print("\n====================================")
        print("Friday Performance Summary")
        print(f"Average STT        : {_fmt_ms(stt)}")
        print(f"Average LLM        : {_fmt_ms(llm)}")
        print(f"Average TTS        : {_fmt_ms(tts)}")
        print(f"Average Playback   : {_fmt_ms(playback)}")
        print("------------------------------------")
        print(f"Average End-to-End : {_fmt_s(end_to_end)}")
        print("====================================")

    def export_json(self, file_path: str = "metrics.json") -> None:
        """
        Writes aggregated stats and raw latency arrays to a JSON file.
        """
        with self._lock:
            data = {
                "summary": {
                    name: (sum(vals) / len(vals) if vals else 0.0)
                    for name, vals in self.metrics.items()
                },
                "raw": self.metrics,
                "wer_average": self.wer_tracker.average,
            }
            
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
        except Exception:
            pass

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Successfully exported latency metrics registry to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write metrics data log to {file_path}: {e}")
