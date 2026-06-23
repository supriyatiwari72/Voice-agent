# Production-Grade Modular Voice-to-Voice AI Agent (MVP V1)

A modular, highly scalable, and provider-agnostic skeleton for a production-grade Voice-to-Voice AI Agent in Python. This framework enforces strict boundary separation between hardware interface threads, signal processors, transcription layers, LLM decision engines, and synthesis backends.

---

## 1. System Architecture & Pipeline Flow

The agent runs a pipeline where incoming microsecond buffers are processed in sequence. The data flow relies on thread-safe queues to isolate high-latency model APIs from real-time audio sample rates.

```
                  [ Audio Capture Loop ]
                            │
                     (Raw Audio Bytes)
                            ▼
                    ┌───────────────┐
                    │  Microphone   │
                    └───────┬───────┘
                            │ (input_buffer)
                            ▼
                    ┌───────────────┐
                    │  Noise Filter │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  VAD Gating   │
                    └───────┬───────┘
                            │ (Voice Segment Boundaries)
                            ▼
                    ┌───────────────┐
                    │  STT Engine   │
                    └───────┬───────┘
                            │ (User Text Transcript)
                            ▼
                    ┌───────────────┐
                    │  LLM Brain    │
                    └───────┬───────┘
                            │ (Token Streaming)
                            ▼
                    ┌───────────────┐
                    │  TTS Voice    │
                    └───────┬───────┘
                            │ (output_buffer)
                            ▼
                    ┌───────────────┐
                    │    Speaker    │
                    └───────────────┘
                  [ Audio Playback Loop ]
```

---

## 2. Directory Structure

```
voice-agent/
├── main.py                    # Entry point initializing factories and pipeline managers
├── README.md                  # System architectural blueprint
├── requirements.txt           # Fundamental dependency definitions
├── .env.example               # Configuration blueprint for API keys and environments
│
├── config/
│   ├── __init__.py
│   ├── config.yaml            # Active pipeline steps and audio settings
│   └── models.yaml            # Swappable model names, endpoints, and attributes
│
├── pipeline/
│   ├── __init__.py
│   ├── voice_pipeline.py      # Main pipeline orchestration loop
│   ├── pipeline_manager.py    # Thread coordinator and lifecycle controller
│   └── pipeline_state.py      # Enum tracks states (IDLE, LISTENING, SPEAKING, etc.)
│
├── audio/
│   ├── __init__.py
│   ├── recorder.py            # Hardware microphone input wrapper
│   ├── player.py              # Hardware playback output speaker wrapper
│   └── audio_buffer.py        # Thread-safe audio byte circular queues
│
├── noise/                     # Signal noise suppression adapters
│   ├── __init__.py
│   ├── base.py                # BaseNoiseCanceller contract
│   ├── factory.py             # NoiseFactory registry
│   └── providers/
│
├── vad/                       # Voice Activity Detection (boundary segmenters)
│   ├── __init__.py
│   ├── base.py                # BaseVAD contract
│   ├── factory.py             # VADFactory registry
│   └── providers/
│
├── stt/                       # Speech-to-Text transcription adapters
│   ├── __init__.py
│   ├── base.py                # BaseSTT contract
│   ├── factory.py             # STTFactory registry
│   └── providers/
│
├── llm/                       # LLM decision adapters
│   ├── __init__.py
│   ├── base.py                # BaseLLM contract
│   ├── factory.py             # LLMFactory registry
│   └── providers/
│
├── tts/                       # Text-to-Speech synthesizer adapters
│   ├── __init__.py
│   ├── base.py                # BaseTTS contract
│   ├── factory.py             # TTSFactory registry
│   └── providers/
│
├── metrics/
│   ├── __init__.py
│   └── latency_tracker.py     # Metrics tracker (STT, LLM TTFT, TTS Synthesis)
│
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Centralized logging configs
│   ├── config_loader.py       # Safe YAML reader
│   └── config_validator.py    # Checks providers/keys validity before boot
│
└── tests/                     # Tests covering configs, factories, and pipeline
```

---

## 3. Architecture Design Patterns

### Dependency Inversion Principle (DIP)
The core `VoicePipeline` does not import or instantiate concrete client libraries (like `openai` or `google-genai`). Instead, it communicates strictly using abstract methods declared in the component contract files (e.g., `BaseSTT`, `BaseLLM`). Interfaces dictate what methods must be supported, decoupling the pipeline loop from vendor updates.

### Factory Pattern
Dynamic model swapping is handled via a parameterized Factory Pattern. Each component has a dedicated factory loader (e.g. `LLMFactory`). The factory:
1. Validates the selector parameter against supported adapter classes.
2. Injects required credentials/attributes loaded from `models.yaml`.
3. Returns a class instance matching the standard Interface.

### State Machine Gating
To prepare for interruptions and context changes, the agent maintains an explicit, thread-safe `PipelineState`:
- **`IDLE`**: Ready and waiting.
- **`LISTENING`**: Mic active, buffering frame bytes.
- **`PROCESSING`**: Silence detected, filtering audio signals.
- **`TRANSCRIBING`**: STT converting speech bytes to text.
- **`THINKING`**: LLM generating reply tokens.
- **`SPEAKING`**: Synthesizer playing voice frames to output.
- **`ERROR`**: Caught exceptions, printing diagnostic traces.
- **`STOPPED`**: Gracefully wound down.

---

## 4. How to Integrate a New Provider

To add a new provider (e.g. adding `Deepgram` STT):

1. **Create the adapter**: Inside `stt/providers/`, add a file (e.g., `deepgram_stt.py`). Create a class `DeepgramSTT` inheriting from `BaseSTT`:
   ```python
   from stt.base import BaseSTT

   class DeepgramSTT(BaseSTT):
       def transcribe(self, audio_data: bytes) -> str:
           # Call Deepgram API here
           return "transcribed text"
   ```
2. **Register in Factory**: Add the class mapping to the registry inside `stt/factory.py`.
3. **Configure**: Update `config.yaml` to set `stt.provider` to `"deepgram"` and add any required API keys to `models.yaml` or `.env`.
