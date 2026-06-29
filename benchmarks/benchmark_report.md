# Voice Agent — Benchmark Report

> Generated: 2026-06-24 12:13
> Run `python benchmarks/benchmark_end_to_end.py` to regenerate with fresh measurements.

---

## 1. STT Provider Comparison

_Input: 1 second of 16kHz mono PCM audio_

| Provider | Avg Latency | P50 | P95 | P99 | RAM (MB) |
|----------|-------------|-----|-----|-----|----------|
| DistilWhisper (distil-small.en) | 12968ms | 12848ms | 13317ms | 13364ms | N/A |
| FasterWhisper (tiny) | 3050ms | 3004ms | 4347ms | 4516ms | N/A |
| Dummy (baseline) | 0ms | 0ms | 0ms | 0ms | N/A |

> **Fastest STT**: Dummy (baseline) (0ms avg)

---

## 2. LLM Provider Comparison

_Prompt: "In one sentence, what is the speed of light?"_

| Provider | TTFT Avg | TTFT P50 | TTFT P95 | Total Avg | Tokens/sec | RAM (MB) |
|----------|----------|----------|----------|-----------|------------|----------|
| Qwen 2.5:3b (Ollama) | 3018ms | 3135ms | 3313ms | 5980ms | 3.5 t/s | N/A |
| Phi3 Mini (Ollama) | 2043ms | 2037ms | 2061ms | 2044ms | 10.8 t/s | N/A |
| Groq (llama3-8b-8192) | 0ms | 0ms | 0ms | 0ms | 1025640.9 t/s | N/A |
| Dummy (baseline) | 0ms | 0ms | 0ms | 0ms | 1082474.1 t/s | N/A |

> **Fastest TTFT**: Groq (llama3-8b-8192) (0ms avg)
> **Highest Throughput**: Groq (llama3-8b-8192) (1025640.9 t/s avg)

---

## 3. TTS Provider Comparison

_Input: short conversational sentences (~10 words)_

| Provider | Synthesis Avg | First Chunk Avg | RTF Avg | RAM (MB) |
|----------|---------------|-----------------|---------|----------|
| Kokoro TTS | 10787ms | 8840ms | 3.284 | N/A |
| Kokoro Streaming TTS | 5465ms | 3522ms | 1.442 | N/A |
| Piper Streaming TTS | 0ms | 0ms | 0.000 | N/A |
| Dummy TTS (baseline) | 0ms | 0ms | 0.003 | N/A |

> **Best RTF**: Piper Streaming TTS (RTF=0.000, lower is better)

---

## 4. End-to-End Pipeline Latency

_Full pipeline: Microphone → STT → LLM → TTS → Playback_

| Stack | First Transcript | First LLM Token | First Audio | Total Turnaround | Success |
|-------|-----------------|-----------------|-------------|-----------------|---------|
| Mock Baseline (dummy + dummy + dummy) | 0ms | 61ms | 62ms | 62ms | 100% |
| Demo Stack (distil_whisper + qwen2.5_3b + kokoro) | 9160ms | 16234ms | 21401ms | N/A | 33% |
| Lightweight Stack (faster_whisper + phi3_mini + piper) | 3649ms | 5045ms | 5052ms | 5054ms | 100% |

> **Fastest E2E**: Lightweight Stack (faster_whisper + phi3_mini + piper) (5054ms total turnaround)

---

## 5. Recommendations

### Demo Stack (Recommended Default)
```yaml
active_providers:
  noise: rnnoise
  vad: silero
  stt: distil_whisper    # Best accuracy/latency tradeoff locally
  llm: qwen2.5_3b        # Run: ollama pull qwen2.5:3b
  tts: kokoro            # Best quality local TTS
```

### Lowest Latency Stack
```yaml
active_providers:
  noise: rnnoise
  vad: silero
  stt: faster_whisper    # Faster but less accurate
  llm: groq              # Cloud — extremely low TTFT (<200ms)
  tts: piper_streaming   # Lightweight streaming TTS
```

### Fully Offline Stack (No Internet)
```yaml
active_providers:
  noise: rnnoise
  vad: silero
  stt: distil_whisper
  llm: qwen2.5_3b       # Via Ollama — works offline
  tts: kokoro
```

---

## 6. Setup Instructions

### Install Ollama (for local LLM)
```bash
# Download from: https://ollama.ai
ollama serve                  # Start Ollama server
ollama pull qwen2.5:3b        # Demo default LLM (~2GB)
ollama pull phi3:mini         # Alternative lightweight LLM
```

### Download Kokoro TTS Weights
```bash
mkdir weights
# kokoro-v0_19.onnx + voices.bin from HuggingFace:
# https://huggingface.co/hexgrad/Kokoro-82M
```

### Set API Keys (optional cloud providers)
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY, DEEPGRAM_API_KEY, etc.
```

### Run the Demo
```bash
python voice_agent.py
```