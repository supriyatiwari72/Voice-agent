"""
STT Provider Benchmark
======================
Measures transcription latency and resource usage for all available STT providers.
Skips providers whose models are not downloaded or available.

Output: benchmarks/results/stt_results.json

Usage:
    python benchmarks/benchmark_stt.py
"""
import os
import time
import json
import numpy as np
from typing import Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _get_resource_snapshot() -> Dict[str, float]:
    """Capture current CPU% and RAM usage in MB."""
    if not PSUTIL_AVAILABLE:
        return {"cpu_percent": 0.0, "ram_mb": 0.0}
    proc = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": proc.memory_info().rss / (1024 * 1024),
    }


def run_stt_benchmark(num_iterations: int = 5) -> Dict[str, Any]:
    print("=" * 60)
    print("  STT Provider Benchmark")
    print("=" * 60)

    from stt.factory import STTFactory

    # 1 second of 16kHz 16-bit mono silence (32000 bytes)
    mock_audio = np.zeros(16000, dtype=np.int16).tobytes()

    # Providers to benchmark: (factory_key, display_name)
    providers = [
        ("distil_whisper",   "DistilWhisper (distil-small.en)"),
        ("faster_whisper",   "FasterWhisper (tiny)"),
        ("dummy",            "Dummy (baseline)"),
    ]

    base_config = {
        "models_meta": {
            "stt_providers": {
                "distil_whisper": {
                    "model_size": "distil-small.en",
                    "device": "cpu",
                    "compute_type": "int8",
                    "beam_size": 5,
                },
                "faster_whisper": {
                    "model_size": "tiny",
                    "device": "cpu",
                    "compute_type": "int8",
                    "beam_size": 5,
                },
            }
        }
    }

    results = {}

    for factory_key, display_name in providers:
        print(f"\n  Benchmarking: {display_name}")
        try:
            provider = STTFactory.get_provider(factory_key, base_config)
        except Exception as e:
            print(f"    ⚠ Skipped (init failed): {e}")
            continue

        latencies = []
        baseline = _get_resource_snapshot()

        for i in range(num_iterations):
            start = time.perf_counter()
            try:
                provider.transcribe(mock_audio)
            except Exception as e:
                print(f"    ⚠ Iteration {i+1} failed: {e}")
                continue
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            print(f"    Iteration {i+1}/{num_iterations}: {elapsed:.1f}ms")

        if not latencies:
            print(f"    ⚠ No successful iterations for {display_name}")
            continue

        snapshot = _get_resource_snapshot()
        results[factory_key] = {
            "display_name": display_name,
            "iterations": len(latencies),
            "latency_ms": {
                "average": float(np.mean(latencies)),
                "p50":     float(np.percentile(latencies, 50)),
                "p95":     float(np.percentile(latencies, 95)),
                "p99":     float(np.percentile(latencies, 99)),
                "min":     float(np.min(latencies)),
                "max":     float(np.max(latencies)),
            },
            "resources": {
                "cpu_percent": snapshot["cpu_percent"],
                "ram_mb":      snapshot["ram_mb"],
                "ram_delta_mb": snapshot["ram_mb"] - baseline["ram_mb"],
            },
        }

        print(
            f"    ✓ Avg: {results[factory_key]['latency_ms']['average']:.1f}ms | "
            f"P50: {results[factory_key]['latency_ms']['p50']:.1f}ms | "
            f"P95: {results[factory_key]['latency_ms']['p95']:.1f}ms | "
            f"RAM: {results[factory_key]['resources']['ram_mb']:.0f}MB"
        )

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_stt_benchmark(num_iterations=5)
    out_path = os.path.join(RESULTS_DIR, "stt_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to: {out_path}")
