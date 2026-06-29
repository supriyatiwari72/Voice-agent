"""
LLM Provider Benchmark
======================
Measures Time-To-First-Token (TTFT), total generation time, and resource usage.
Skips providers not available (Ollama not running, API key absent, etc.).

Output: benchmarks/results/llm_results.json

Usage:
    python benchmarks/benchmark_llm.py
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
BENCHMARK_PROMPT = "In one sentence, what is the speed of light?"


def _get_resource_snapshot() -> Dict[str, float]:
    if not PSUTIL_AVAILABLE:
        return {"cpu_percent": 0.0, "ram_mb": 0.0}
    proc = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": proc.memory_info().rss / (1024 * 1024),
    }


def _benchmark_provider(provider, display_name: str, num_iterations: int) -> Dict[str, Any]:
    """Run timing benchmark for a single LLM provider."""
    ttfts = []
    total_times = []
    token_counts = []
    baseline = _get_resource_snapshot()

    for i in range(num_iterations):
        start = time.perf_counter()
        try:
            stream = provider.generate_stream(BENCHMARK_PROMPT)
            first_token = next(stream, None)
            if first_token is None:
                print(f"    ⚠ Iteration {i+1}: No tokens returned")
                continue

            ttft = (time.perf_counter() - start) * 1000
            ttfts.append(ttft)

            # Drain remaining tokens
            token_count = 1
            for _ in stream:
                token_count += 1

            total = (time.perf_counter() - start) * 1000
            total_times.append(total)
            token_counts.append(token_count)

            tps = token_count / (total / 1000) if total > 0 else 0
            print(
                f"    Iteration {i+1}/{num_iterations}: "
                f"TTFT={ttft:.0f}ms | Total={total:.0f}ms | "
                f"Tokens={token_count} | TPS={tps:.1f}"
            )
        except Exception as e:
            print(f"    ⚠ Iteration {i+1} failed: {e}")
            continue

    if not ttfts:
        return None

    snapshot = _get_resource_snapshot()
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0
    avg_total = float(np.mean(total_times)) if total_times else 0
    tps = avg_tokens / (avg_total / 1000) if avg_total > 0 else 0

    return {
        "display_name": display_name,
        "iterations": len(ttfts),
        "ttft_ms": {
            "average": float(np.mean(ttfts)),
            "p50":     float(np.percentile(ttfts, 50)),
            "p95":     float(np.percentile(ttfts, 95)),
            "min":     float(np.min(ttfts)),
        },
        "total_ms": {
            "average": avg_total,
            "p50":     float(np.percentile(total_times, 50)) if total_times else 0,
            "p95":     float(np.percentile(total_times, 95)) if total_times else 0,
        },
        "tokens_per_sec": tps,
        "avg_token_count": avg_tokens,
        "resources": {
            "cpu_percent": snapshot["cpu_percent"],
            "ram_mb":      snapshot["ram_mb"],
            "ram_delta_mb": snapshot["ram_mb"] - baseline["ram_mb"],
        },
    }


def run_llm_benchmark(num_iterations: int = 3) -> Dict[str, Any]:
    print("=" * 60)
    print("  LLM Provider Benchmark")
    print("=" * 60)

    from llm.factory import LLMFactory

    # (factory_key, display_name)
    # Ollama providers require `ollama serve` and the model pulled.
    providers = [
        ("qwen2.5_3b",  "Qwen 2.5:3b (Ollama)"),
        ("phi3_mini",   "Phi3 Mini (Ollama)"),
        ("groq",        "Groq (llama3-8b-8192)"),
        ("dummy",       "Dummy (baseline)"),
    ]

    # Build a minimal config with models_meta
    import yaml, os as _os
    config = {}
    try:
        with open(_os.path.join(_os.path.dirname(__file__), "..", "config", "config.yaml"), encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        with open(_os.path.join(_os.path.dirname(__file__), "..", "config", "models.yaml"), encoding="utf-8") as f:
            config["models_meta"] = yaml.safe_load(f) or {}
    except Exception:
        pass  # Proceed with empty config — providers handle missing config gracefully

    results = {}

    for factory_key, display_name in providers:
        print(f"\n  Benchmarking: {display_name}")
        try:
            provider = LLMFactory.get_provider(factory_key, config)
        except Exception as e:
            print(f"    ⚠ Skipped (init failed): {e}")
            continue

        result = _benchmark_provider(provider, display_name, num_iterations)
        if result is None:
            print(f"    ⚠ No successful iterations for {display_name}")
            continue

        results[factory_key] = result
        print(
            f"    ✓ TTFT Avg: {result['ttft_ms']['average']:.0f}ms | "
            f"TPS: {result['tokens_per_sec']:.1f} | "
            f"RAM: {result['resources']['ram_mb']:.0f}MB"
        )

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_llm_benchmark(num_iterations=3)
    out_path = os.path.join(RESULTS_DIR, "llm_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to: {out_path}")
