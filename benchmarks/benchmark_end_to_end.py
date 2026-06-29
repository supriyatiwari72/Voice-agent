"""
End-to-End Pipeline Benchmark
==============================
Runs full pipeline permutations and collects all 5 streaming latency metrics
plus CPU/RAM resource usage.

Provider stacks benchmarked:
  Stack A (Demo Default): distil_whisper + qwen2.5_3b + kokoro
  Stack B (Lightweight):  faster_whisper + phi3_mini + piper_streaming
  Stack C (Mock Baseline): dummy + dummy + dummy

Output: benchmarks/results/e2e_results.json

Usage:
    python benchmarks/benchmark_end_to_end.py
"""
import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List
from unittest.mock import MagicMock

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add project root to path so imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _get_resource_snapshot() -> Dict[str, float]:
    if not PSUTIL_AVAILABLE:
        return {"cpu_percent": 0.0, "ram_mb": 0.0}
    proc = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": proc.memory_info().rss / (1024 * 1024),
    }


def _run_pipeline_iteration(config: Dict[str, Any], speech_pcm: bytes, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Run one full pipeline iteration and collect metrics.
    Returns metric dict or raises on timeout.
    """
    from pipeline.pipeline_manager import PipelineManager

    # Force VAD min_speech_bytes to 0 for the benchmark run so that short trigger inputs
    # are never discarded by the pipeline gating logic.
    if "models_meta" not in config:
        config["models_meta"] = {}
    if "vad_providers" not in config["models_meta"]:
        config["models_meta"]["vad_providers"] = {}
    if "silero" not in config["models_meta"]["vad_providers"]:
        config["models_meta"]["vad_providers"]["silero"] = {}
    config["models_meta"]["vad_providers"]["silero"]["min_speech_bytes"] = 0
    config["models_meta"]["vad_providers"]["silero"]["max_silence_frames"] = 10

    manager = PipelineManager(config)
    manager.initialize_pipeline()

    # Mock the audio player to prevent hardware output during benchmarking
    manager.player.start_playback = MagicMock()

    manager.start()
    metrics = {}

    try:
        # Push voice/trigger audio chunks to trigger VAD
        if speech_pcm and config.get("active_providers", {}).get("vad") == "silero":
            print(" [Pushing real voice PCM...]", end="", flush=True)
            for i in range(0, len(speech_pcm), 960):
                chunk = speech_pcm[i:i+960]
                if len(chunk) < 960:
                    chunk = chunk + b"\x00" * (960 - len(chunk))
                manager.input_buffer.push(chunk)
                time.sleep(0.005)
        else:
            # Pushing a mock speech frame
            manager.input_buffer.push(b"\x01\x02" * 480)
            time.sleep(0.01)

        # Push silence frames to trigger VAD speech boundary completion
        for _ in range(15):
            manager.input_buffer.push(b"\x00" * 960)
            time.sleep(0.01)

        # Wait for output audio to arrive in playback buffer
        start_wait = time.perf_counter()
        completed = False
        while (time.perf_counter() - start_wait) < timeout:
            if manager.output_buffer.size() > 0:
                completed = True
                break
            time.sleep(0.1)

        # Allow pipeline to settle for final metric recording
        time.sleep(0.2)

        summary = manager.metrics_tracker.get_summary()
        for key in [
            "first_partial_transcript_ms",
            "first_llm_token_ms",
            "first_sentence_ms",
            "first_audio_chunk_ms",
            "total_turnaround_ms",
            "stt_latency_ms",
            "llm_latency_ms",
            "tts_latency_ms",
            "ttft_ms",
        ]:
            if key in summary and summary[key].get("count", 0) > 0:
                metrics[key] = summary[key]["average"]

        metrics["completed"] = completed

    finally:
        manager.stop()

    return metrics


def run_e2e_benchmark(num_iterations: int = 3) -> Dict[str, Any]:
    print("=" * 60)
    print("  End-to-End Pipeline Benchmark")
    print("=" * 60)

    import yaml

    # Load config from project with explicit UTF-8 encoding
    config = {}
    try:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
        with open(os.path.join(config_dir, "config.yaml"), encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        with open(os.path.join(config_dir, "models.yaml"), encoding="utf-8") as f:
            config["models_meta"] = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  ⚠ Could not load config: {e}. Using defaults.")

    # Ensure queues are configured
    config.setdefault("queues", {
        "audio_queue_size": 200,
        "speech_queue_size": 100,
        "transcript_queue_size": 20,
        "response_queue_size": 20,
        "tts_queue_size": 20,
        "playback_queue_size": 50,
    })
    config.setdefault("audio", {"buffer_max_size": 200})

    # Pre-generate real speech PCM using Kokoro so VAD is triggered correctly
    speech_pcm = b""
    try:
        from tts.factory import TTSFactory
        kokoro_config = {
            "models_meta": {
                "tts_providers": {
                    "kokoro": {
                        "model_path": "weights/kokoro-v0_19.onnx",
                        "voices_path": "weights/voices.bin",
                        "voice": "af_bella"
                    }
                }
            }
        }
        print("  Generating real voice sample for VAD triggers...")
        tts = TTSFactory.get_provider("kokoro", kokoro_config)
        speech_pcm = tts.synthesize("Hello")
        print(f"  ✓ Generated voice sample: {len(speech_pcm)} bytes PCM")
    except Exception as e:
        print(f"  ⚠ Could not pre-generate voice sample: {e}. Falling back to default frames.")

    # Stack definitions
    stacks = [
        {
            "name": "mock_baseline",
            "display": "Mock Baseline (dummy + dummy + dummy)",
            "providers": {"noise": "dummy", "vad": "dummy", "stt": "dummy", "llm": "dummy", "tts": "dummy"},
        },
        {
            "name": "demo_stack",
            "display": "Demo Stack (distil_whisper + qwen2.5_3b + kokoro)",
            "providers": {"noise": "rnnoise", "vad": "silero", "stt": "distil_whisper", "llm": "qwen2.5_3b", "tts": "kokoro"},
        },
        {
            "name": "lightweight_stack",
            "display": "Lightweight Stack (faster_whisper + phi3_mini + piper)",
            "providers": {"noise": "rnnoise", "vad": "silero", "stt": "faster_whisper", "llm": "phi3_mini", "tts": "piper_streaming"},
        },
    ]

    results = {}

    for stack in stacks:
        print(f"\n  Stack: {stack['display']}")
        stack_config = dict(config)
        stack_config["active_providers"] = stack["providers"]

        iteration_metrics: List[Dict[str, float]] = []
        baseline = _get_resource_snapshot()

        for i in range(num_iterations):
            print(f"    Iteration {i+1}/{num_iterations}...", end=" ", flush=True)
            try:
                # Use a larger timeout for real stack on CPU
                is_real_stack = (stack["name"] != "mock_baseline")
                m = _run_pipeline_iteration(stack_config, speech_pcm, timeout=25.0 if is_real_stack else 5.0)
                iteration_metrics.append(m)
                turnaround = m.get("total_turnaround_ms", 0)
                ttft = m.get("first_llm_token_ms", 0)
                audio = m.get("first_audio_chunk_ms", 0)
                print(f"Turnaround={turnaround:.0f}ms | TTFT={ttft:.0f}ms | FirstAudio={audio:.0f}ms")
            except Exception as e:
                print(f"⚠ Failed: {e}")
                continue

        if not iteration_metrics:
            print(f"    ⚠ No successful iterations for {stack['display']}")
            continue

        snapshot = _get_resource_snapshot()

        def avg(key: str) -> float:
            vals = [m[key] for m in iteration_metrics if key in m and m[key] > 0]
            return float(np.mean(vals)) if vals else 0.0

        def p50(key: str) -> float:
            vals = [m[key] for m in iteration_metrics if key in m and m[key] > 0]
            return float(np.percentile(vals, 50)) if vals else 0.0

        def p95(key: str) -> float:
            vals = [m[key] for m in iteration_metrics if key in m and m[key] > 0]
            return float(np.percentile(vals, 95)) if vals else 0.0

        results[stack["name"]] = {
            "display_name": stack["display"],
            "iterations": len(iteration_metrics),
            "success_rate": sum(1 for m in iteration_metrics if m.get("completed")) / len(iteration_metrics),
            "latency_ms": {
                "first_transcript":    {"avg": avg("first_partial_transcript_ms"), "p50": p50("first_partial_transcript_ms"), "p95": p95("first_partial_transcript_ms")},
                "first_llm_token":     {"avg": avg("first_llm_token_ms"),          "p50": p50("first_llm_token_ms"),          "p95": p95("first_llm_token_ms")},
                "first_sentence":      {"avg": avg("first_sentence_ms"),            "p50": p50("first_sentence_ms"),            "p95": p95("first_sentence_ms")},
                "first_audio_chunk":   {"avg": avg("first_audio_chunk_ms"),         "p50": p50("first_audio_chunk_ms"),         "p95": p95("first_audio_chunk_ms")},
                "total_turnaround":    {"avg": avg("total_turnaround_ms"),          "p50": p50("total_turnaround_ms"),          "p95": p95("total_turnaround_ms")},
            },
            "resources": {
                "cpu_percent":  snapshot["cpu_percent"],
                "ram_mb":       snapshot["ram_mb"],
                "ram_delta_mb": snapshot["ram_mb"] - baseline["ram_mb"],
            },
        }

        print(
            f"    ✓ Total Turnaround Avg: {results[stack['name']]['latency_ms']['total_turnaround']['avg']:.0f}ms | "
            f"Success Rate: {results[stack['name']]['success_rate']*100:.0f}%"
        )

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_e2e_benchmark(num_iterations=3)
    out_path = os.path.join(RESULTS_DIR, "e2e_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to: {out_path}")
    print("\n  Run 'python benchmarks/report_generator.py' to generate benchmark_report.md")
