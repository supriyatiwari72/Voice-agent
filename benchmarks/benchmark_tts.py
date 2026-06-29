"""
TTS Provider Benchmark
======================
Measures synthesis latency, Real-Time Factor (RTF), and resource usage.

RTF = synthesis_time / audio_duration
RTF < 1.0 means faster than real-time (good for streaming).

Output: benchmarks/results/tts_results.json

Usage:
    python benchmarks/benchmark_tts.py
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

# Short test sentence — representative of voice agent response length
TEST_SENTENCES = [
    "Hello, how can I help you today?",
    "The speed of light is approximately 299 million meters per second.",
    "Sure, I can help with that.",
]


def _get_resource_snapshot() -> Dict[str, float]:
    if not PSUTIL_AVAILABLE:
        return {"cpu_percent": 0.0, "ram_mb": 0.0}
    proc = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": proc.memory_info().rss / (1024 * 1024),
    }


def _estimate_audio_duration(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, bit_depth: int = 16) -> float:
    """Estimate audio duration from raw PCM bytes in seconds."""
    bytes_per_sample = bit_depth // 8
    num_samples = len(pcm_bytes) / (bytes_per_sample * channels)
    return num_samples / sample_rate


def run_tts_benchmark(num_iterations: int = 3) -> Dict[str, Any]:
    print("=" * 60)
    print("  TTS Provider Benchmark")
    print("=" * 60)

    from tts.factory import TTSFactory

    providers = [
        ("kokoro",          "Kokoro TTS"),
        ("kokoro_streaming","Kokoro Streaming TTS"),
        ("piper_streaming", "Piper Streaming TTS"),
        ("dummy",           "Dummy TTS (baseline)"),
    ]

    import yaml, os as _os
    config = {}
    try:
        with open(_os.path.join(_os.path.dirname(__file__), "..", "config", "config.yaml"), encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        with open(_os.path.join(_os.path.dirname(__file__), "..", "config", "models.yaml"), encoding="utf-8") as f:
            config["models_meta"] = yaml.safe_load(f) or {}
    except Exception:
        pass

    results = {}

    for factory_key, display_name in providers:
        print(f"\n  Benchmarking: {display_name}")
        try:
            provider = TTSFactory.get_provider(factory_key, config)
        except Exception as e:
            print(f"    ⚠ Skipped (init failed): {e}")
            continue

        latencies = []
        rtfs = []
        chunk_first_latencies = []
        baseline = _get_resource_snapshot()

        for i in range(num_iterations):
            test_text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            start = time.perf_counter()

            try:
                # Test streaming synthesis
                if hasattr(provider, "stream_synthesize"):
                    first_chunk_time = None
                    all_audio = b""
                    for chunk in provider.stream_synthesize(test_text):
                        if first_chunk_time is None:
                            first_chunk_time = (time.perf_counter() - start) * 1000
                        all_audio += chunk
                    elapsed = (time.perf_counter() - start) * 1000
                elif hasattr(provider, "synthesize_stream"):
                    first_chunk_time = None
                    all_audio = b""
                    for chunk in provider.synthesize_stream(test_text):
                        if first_chunk_time is None:
                            first_chunk_time = (time.perf_counter() - start) * 1000
                        all_audio += chunk
                    elapsed = (time.perf_counter() - start) * 1000
                else:
                    all_audio = provider.synthesize(test_text)
                    elapsed = (time.perf_counter() - start) * 1000
                    first_chunk_time = elapsed

                audio_dur = _estimate_audio_duration(all_audio)
                rtf = (elapsed / 1000) / audio_dur if audio_dur > 0 else 0

                latencies.append(elapsed)
                rtfs.append(rtf)
                if first_chunk_time is not None:
                    chunk_first_latencies.append(first_chunk_time)

                print(
                    f"    Iteration {i+1}/{num_iterations}: "
                    f"Synthesis={elapsed:.0f}ms | "
                    f"First Chunk={first_chunk_time:.0f}ms | "
                    f"RTF={rtf:.3f} | "
                    f"Audio={audio_dur:.2f}s"
                )

            except Exception as e:
                print(f"    ⚠ Iteration {i+1} failed: {e}")
                continue

        if not latencies:
            continue

        snapshot = _get_resource_snapshot()
        results[factory_key] = {
            "display_name": display_name,
            "iterations": len(latencies),
            "synthesis_ms": {
                "average": float(np.mean(latencies)),
                "p50":     float(np.percentile(latencies, 50)),
                "p95":     float(np.percentile(latencies, 95)),
                "min":     float(np.min(latencies)),
            },
            "first_chunk_ms": {
                "average": float(np.mean(chunk_first_latencies)) if chunk_first_latencies else 0,
                "p50":     float(np.percentile(chunk_first_latencies, 50)) if chunk_first_latencies else 0,
            },
            "rtf": {
                "average": float(np.mean(rtfs)),
                "p50":     float(np.percentile(rtfs, 50)),
                "min":     float(np.min(rtfs)),
            },
            "resources": {
                "cpu_percent": snapshot["cpu_percent"],
                "ram_mb":      snapshot["ram_mb"],
                "ram_delta_mb": snapshot["ram_mb"] - baseline["ram_mb"],
            },
        }

        print(
            f"    ✓ Synthesis Avg: {results[factory_key]['synthesis_ms']['average']:.0f}ms | "
            f"First Chunk: {results[factory_key]['first_chunk_ms']['average']:.0f}ms | "
            f"RTF: {results[factory_key]['rtf']['average']:.3f}"
        )

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_tts_benchmark(num_iterations=3)
    out_path = os.path.join(RESULTS_DIR, "tts_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to: {out_path}")
