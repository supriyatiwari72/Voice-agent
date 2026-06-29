"""
Benchmark Report Generator
===========================
Reads all JSON result files from benchmarks/results/ and generates
a comprehensive benchmark_report.md with comparison tables and recommendations.

Usage:
    python benchmarks/report_generator.py
    # Output: benchmarks/benchmark_report.md
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_report.md")


def _load_json(filename: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Could not load {filename}: {e}")
        return None


def _fmt(value: float, unit: str = "ms", decimals: int = 0) -> str:
    if value == 0:
        return "N/A"
    return f"{value:.{decimals}f}{unit}"


def _generate_stt_table(data: Dict[str, Any]) -> str:
    if not data:
        return "_No STT benchmark data available. Run `python benchmarks/benchmark_stt.py`_\n"

    lines = [
        "| Provider | Avg Latency | P50 | P95 | P99 | RAM (MB) |",
        "|----------|-------------|-----|-----|-----|----------|",
    ]
    fastest = None
    fastest_lat = float("inf")

    for key, r in data.items():
        lat = r.get("latency_ms", {})
        res = r.get("resources", {})
        avg = lat.get("average", 0)
        lines.append(
            f"| {r.get('display_name', key)} "
            f"| {_fmt(avg)} "
            f"| {_fmt(lat.get('p50', 0))} "
            f"| {_fmt(lat.get('p95', 0))} "
            f"| {_fmt(lat.get('p99', 0))} "
            f"| {_fmt(res.get('ram_mb', 0), 'MB')} |"
        )
        if avg > 0 and avg < fastest_lat:
            fastest_lat = avg
            fastest = r.get("display_name", key)

    lines.append("")
    if fastest:
        lines.append(f"> **Fastest STT**: {fastest} ({_fmt(fastest_lat)} avg)")

    return "\n".join(lines)


def _generate_llm_table(data: Dict[str, Any]) -> str:
    if not data:
        return "_No LLM benchmark data available. Run `python benchmarks/benchmark_llm.py`_\n"

    lines = [
        "| Provider | TTFT Avg | TTFT P50 | TTFT P95 | Total Avg | Tokens/sec | RAM (MB) |",
        "|----------|----------|----------|----------|-----------|------------|----------|",
    ]
    fastest_ttft = None
    fastest_val = float("inf")
    highest_tps = None
    highest_tps_val = 0.0

    for key, r in data.items():
        ttft = r.get("ttft_ms", {})
        total = r.get("total_ms", {})
        res = r.get("resources", {})
        tps = r.get("tokens_per_sec", 0)
        avg_ttft = ttft.get("average", 0)

        lines.append(
            f"| {r.get('display_name', key)} "
            f"| {_fmt(avg_ttft)} "
            f"| {_fmt(ttft.get('p50', 0))} "
            f"| {_fmt(ttft.get('p95', 0))} "
            f"| {_fmt(total.get('average', 0))} "
            f"| {_fmt(tps, ' t/s', 1)} "
            f"| {_fmt(res.get('ram_mb', 0), 'MB')} |"
        )

        if avg_ttft > 0 and avg_ttft < fastest_val and key != "dummy":
            fastest_val = avg_ttft
            fastest_ttft = r.get("display_name", key)
        if tps > highest_tps_val and key != "dummy":
            highest_tps_val = tps
            highest_tps = r.get("display_name", key)

    lines.append("")
    if fastest_ttft:
        lines.append(f"> **Fastest TTFT**: {fastest_ttft} ({_fmt(fastest_val)} avg)")
    if highest_tps:
        lines.append(f"> **Highest Throughput**: {highest_tps} ({_fmt(highest_tps_val, ' t/s', 1)} avg)")

    return "\n".join(lines)


def _generate_tts_table(data: Dict[str, Any]) -> str:
    if not data:
        return "_No TTS benchmark data available. Run `python benchmarks/benchmark_tts.py`_\n"

    lines = [
        "| Provider | Synthesis Avg | First Chunk Avg | RTF Avg | RAM (MB) |",
        "|----------|---------------|-----------------|---------|----------|",
    ]
    best_rtf = None
    best_rtf_val = float("inf")

    for key, r in data.items():
        synth = r.get("synthesis_ms", {})
        chunk = r.get("first_chunk_ms", {})
        rtf = r.get("rtf", {})
        res = r.get("resources", {})
        rtf_avg = rtf.get("average", 0)

        lines.append(
            f"| {r.get('display_name', key)} "
            f"| {_fmt(synth.get('average', 0))} "
            f"| {_fmt(chunk.get('average', 0))} "
            f"| {_fmt(rtf_avg, '', 3)} "
            f"| {_fmt(res.get('ram_mb', 0), 'MB')} |"
        )

        if 0 < rtf_avg < best_rtf_val and key != "dummy":
            best_rtf_val = rtf_avg
            best_rtf = r.get("display_name", key)

    lines.append("")
    if best_rtf:
        lines.append(f"> **Best RTF**: {best_rtf} (RTF={_fmt(best_rtf_val, '', 3)}, lower is better)")

    return "\n".join(lines)


def _generate_e2e_table(data: Dict[str, Any]) -> str:
    if not data:
        return "_No E2E benchmark data available. Run `python benchmarks/benchmark_end_to_end.py`_\n"

    lines = [
        "| Stack | First Transcript | First LLM Token | First Audio | Total Turnaround | Success |",
        "|-------|-----------------|-----------------|-------------|-----------------|---------|",
    ]
    best_stack = None
    best_turnaround = float("inf")
    lowest_ram = None
    lowest_ram_val = float("inf")

    for key, r in data.items():
        lat = r.get("latency_ms", {})
        res = r.get("resources", {})
        turnaround = lat.get("total_turnaround", {}).get("avg", 0)
        success = r.get("success_rate", 0) * 100
        ram = res.get("ram_mb", 0)

        lines.append(
            f"| {r.get('display_name', key)} "
            f"| {_fmt(lat.get('first_transcript', {}).get('avg', 0))} "
            f"| {_fmt(lat.get('first_llm_token', {}).get('avg', 0))} "
            f"| {_fmt(lat.get('first_audio_chunk', {}).get('avg', 0))} "
            f"| {_fmt(turnaround)} "
            f"| {success:.0f}% |"
        )

        if turnaround > 0 and turnaround < best_turnaround and key != "mock_baseline":
            best_turnaround = turnaround
            best_stack = r.get("display_name", key)
        if ram > 0 and ram < lowest_ram_val and key != "mock_baseline":
            lowest_ram_val = ram
            lowest_ram = r.get("display_name", key)

    lines.append("")
    if best_stack:
        lines.append(f"> **Fastest E2E**: {best_stack} ({_fmt(best_turnaround)} total turnaround)")
    if lowest_ram:
        lines.append(f"> **Lowest Memory**: {lowest_ram} ({_fmt(lowest_ram_val, 'MB')} RAM)")

    return "\n".join(lines)


def generate_report() -> None:
    print("=" * 60)
    print("  Benchmark Report Generator")
    print("=" * 60)

    stt_data = _load_json("stt_results.json")
    llm_data = _load_json("llm_results.json")
    tts_data = _load_json("tts_results.json")
    e2e_data = _load_json("e2e_results.json")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Voice Agent — Benchmark Report",
        "",
        f"> Generated: {timestamp}",
        "> Run `python benchmarks/benchmark_end_to_end.py` to regenerate with fresh measurements.",
        "",
        "---",
        "",
        "## 1. STT Provider Comparison",
        "",
        "_Input: 1 second of 16kHz mono PCM audio_",
        "",
        _generate_stt_table(stt_data or {}),
        "",
        "---",
        "",
        "## 2. LLM Provider Comparison",
        "",
        f'_Prompt: "In one sentence, what is the speed of light?"_',
        "",
        _generate_llm_table(llm_data or {}),
        "",
        "---",
        "",
        "## 3. TTS Provider Comparison",
        "",
        "_Input: short conversational sentences (~10 words)_",
        "",
        _generate_tts_table(tts_data or {}),
        "",
        "---",
        "",
        "## 4. End-to-End Pipeline Latency",
        "",
        "_Full pipeline: Microphone → STT → LLM → TTS → Playback_",
        "",
        _generate_e2e_table(e2e_data or {}),
        "",
        "---",
        "",
        "## 5. Recommendations",
        "",
        "### Demo Stack (Recommended Default)",
        "```yaml",
        "active_providers:",
        "  noise: rnnoise",
        "  vad: silero",
        "  stt: distil_whisper    # Best accuracy/latency tradeoff locally",
        "  llm: qwen2.5_3b        # Run: ollama pull qwen2.5:3b",
        "  tts: kokoro            # Best quality local TTS",
        "```",
        "",
        "### Lowest Latency Stack",
        "```yaml",
        "active_providers:",
        "  noise: rnnoise",
        "  vad: silero",
        "  stt: faster_whisper    # Faster but less accurate",
        "  llm: groq              # Cloud — extremely low TTFT (<200ms)",
        "  tts: piper_streaming   # Lightweight streaming TTS",
        "```",
        "",
        "### Fully Offline Stack (No Internet)",
        "```yaml",
        "active_providers:",
        "  noise: rnnoise",
        "  vad: silero",
        "  stt: distil_whisper",
        "  llm: qwen2.5_3b       # Via Ollama — works offline",
        "  tts: kokoro",
        "```",
        "",
        "---",
        "",
        "## 6. Setup Instructions",
        "",
        "### Install Ollama (for local LLM)",
        "```bash",
        "# Download from: https://ollama.ai",
        "ollama serve                  # Start Ollama server",
        "ollama pull qwen2.5:3b        # Demo default LLM (~2GB)",
        "ollama pull phi3:mini         # Alternative lightweight LLM",
        "```",
        "",
        "### Download Kokoro TTS Weights",
        "```bash",
        "mkdir weights",
        "# kokoro-v0_19.onnx + voices.bin from HuggingFace:",
        "# https://huggingface.co/hexgrad/Kokoro-82M",
        "```",
        "",
        "### Set API Keys (optional cloud providers)",
        "```bash",
        "cp .env.example .env",
        "# Edit .env and set GROQ_API_KEY, DEEPGRAM_API_KEY, etc.",
        "```",
        "",
        "### Run the Demo",
        "```bash",
        "python voice_agent.py",
        "```",
    ]

    report = "\n".join(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  ✓ Report written to: {REPORT_PATH}")
    print(f"  Sections: STT | LLM | TTS | E2E | Recommendations | Setup")


if __name__ == "__main__":
    generate_report()
