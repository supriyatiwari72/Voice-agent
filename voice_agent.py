"""
voice_agent.py — Interactive Voice Agent Demo
==============================================
Real-time Voice-to-Voice AI Agent demo runner.

Pipeline:
    Microphone → RNNoise → Silero VAD → DistilWhisper → Qwen 2.5:3b → Kokoro → Speakers

Usage:
    python voice_agent.py              # Uses config/config.yaml (demo defaults)
    python voice_agent.py --list       # List available providers

Requirements:
    - Ollama running locally for LLM:   ollama serve && ollama pull qwen2.5:3b
    - Kokoro model files in weights/    (kokoro-v0_19.onnx + voices.bin)
    - Microphone and speakers connected

Press Ctrl+C to quit.
"""
import argparse
import logging
import os
import signal
import sys
import time
import threading
from typing import Optional

# Suppress benign Hugging Face Hub warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv()

from utils.config_loader import ConfigLoader
from utils.config_validator import ConfigValidator
from utils.logger import setup_logger
from pipeline.pipeline_manager import PipelineManager
from core.events import EventType

logger = logging.getLogger("voice_agent")


# ANSI color codes for console output
class Colors:
    RESET  = "\033[0m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    MAGENTA = "\033[35m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"
    RED    = "\033[31m"


def _c(text: str, color: str) -> str:
    """Wrap text with color if stdout supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


def print_banner(config: dict) -> None:
    providers = config.get("active_providers", {})
    print()
    print(_c("+----------------------------------------------+", Colors.CYAN))
    print(_c("|         Voice-to-Voice AI Agent Demo         |", Colors.CYAN))
    print(_c("+----------------------------------------------+", Colors.CYAN))
    print()
    print(_c("  Active Providers:", Colors.BOLD))
    print(f"  {'Noise':10s} {_c(providers.get('noise', 'N/A'), Colors.GREEN)}")
    print(f"  {'VAD':10s} {_c(providers.get('vad', 'N/A'), Colors.GREEN)}")
    print(f"  {'STT':10s} {_c(providers.get('stt', 'N/A'), Colors.GREEN)}")
    print(f"  {'LLM':10s} {_c(providers.get('llm', 'N/A'), Colors.GREEN)}")
    print(f"  {'TTS':10s} {_c(providers.get('tts', 'N/A'), Colors.GREEN)}")
    print()
    print(_c("  Press Ctrl+C to quit.", Colors.DIM))
    print()


def _format_latency(ms: float) -> str:
    if ms <= 0:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.1f}s"


from utils.popup import StatusPopup
from pipeline.pipeline_state import PipelineState


class ConsolePipelineListener:
    """
    Hooks into PipelineContext event callbacks to track conversation state
    without terminal log pollution.
    """

    def __init__(self, context, metrics_tracker):
        self.context = context
        self.metrics_tracker = metrics_tracker

    def register(self) -> None:
        """Register event listeners on the pipeline context."""
        ctx = self.context
        ctx.register_event_listener(EventType.SPEECH_STARTED,        self._on_speech_start)
        ctx.register_event_listener(EventType.SPEECH_ENDED,          self._on_speech_end)
        ctx.register_event_listener(EventType.INTERRUPTION_STARTED,  self._on_interrupt)
        ctx.register_event_listener(EventType.INTERRUPTION_FINISHED, self._on_interrupt_done)
        ctx.register_state_listener(self._on_state_changed)

    def _on_speech_start(self, req_id: str = "") -> None:
        logger.info(f"Speech start event observed: {req_id}")

    def _on_speech_end(self, req_id: str = "") -> None:
        logger.info(f"Speech end event observed: {req_id}")

    def _on_interrupt(self, req_id: str = "") -> None:
        logger.warning(f"Barge-in interruption event observed: {req_id}")

    def _on_interrupt_done(self, req_id: str = "") -> None:
        logger.info("Barge-in interruption finished.")

    def _on_state_changed(self, state) -> None:
        from pipeline.pipeline_state import PipelineState
        if state == PipelineState.IDLE:
            print("\n[Speak Now...]\n", flush=True)
        elif state == PipelineState.LISTENING:
            print("\n[Listening...]\n", flush=True)
        elif state == PipelineState.PROCESSING:
            print("\n[Understanding...]\n", flush=True)
        elif state == PipelineState.THINKING:
            print("\n[Thinking...]\n", flush=True)
        elif state == PipelineState.SPEAKING:
            pass


def list_providers() -> None:
    """Print all registered providers across all factories."""
    from stt.factory import STTFactory
    from llm.factory import LLMFactory
    from tts.factory import TTSFactory

    print("\nAvailable Providers:")
    print(f"  STT: {', '.join(STTFactory._providers.keys())}")
    print(f"  LLM: {', '.join(LLMFactory._providers.keys())}")
    print(f"  TTS: {', '.join(TTSFactory._providers.keys())}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice-to-Voice AI Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available providers and exit."
    )
    parser.add_argument(
        "--stt", type=str, default=None,
        help="Override STT provider (e.g. distil_whisper, faster_whisper)"
    )
    parser.add_argument(
        "--llm", type=str, default=None,
        help="Override LLM provider (e.g. qwen2.5_3b, phi3_mini, groq)"
    )
    parser.add_argument(
        "--tts", type=str, default=None,
        help="Override TTS provider (e.g. kokoro, piper_streaming)"
    )
    args = parser.parse_args()

    if args.list:
        list_providers()
        return

    # ── Load Config ────────────────────────────────────────────────────────
    try:
        config = ConfigLoader.load_yaml("config/config.yaml")
        models = ConfigLoader.load_yaml("config/models.yaml")
        config["models_meta"] = models
    except Exception as e:
        print(f"[Error] Failed to load configuration: {e}")
        sys.exit(1)

    # Apply CLI overrides
    providers = config.setdefault("active_providers", {})
    if args.stt:
        providers["stt"] = args.stt
    if args.llm:
        providers["llm"] = args.llm
    if args.tts:
        providers["tts"] = args.tts

    # Ensure queues config exists
    config.setdefault("queues", {
        "audio_queue_size": 200,
        "speech_queue_size": 100,
        "transcript_queue_size": 20,
        "response_queue_size": 20,
        "tts_queue_size": 20,
        "playback_queue_size": 50,
    })

    # ── Setup Logging ──────────────────────────────────────────────────────
    setup_logger(config)

    # Suppress verbose library logging during demo
    for noisy in ["faster_whisper", "ctranslate2", "torch", "urllib3", "httpx", "huggingface_hub"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Validate Config ────────────────────────────────────────────────────
    try:
        ConfigValidator.validate(config)
    except Exception as e:
        print(f"[Error] Configuration validation failed: {e}")
        sys.exit(1)

    # ── Initialize Popup ───────────────────────────────────────────────────
    popup = StatusPopup()
    popup.start()

    import sys

    # ── Initialize Pipeline ────────────────────────────────────────────────
    try:
        manager = PipelineManager(config)
        manager.initialize_pipeline()
    except Exception as e:
        logger.critical(f"Pipeline initialization failed: {e}", exc_info=True)
        popup.destroy()
        sys.exit(1)

    # Wire PTT callback: clicking Speak Now sets ptt_active on the context
    def _ptt_callback():
        current_state = manager.context.get_state()

        # If Friday is speaking, trigger manual interruption immediately
        if current_state == PipelineState.SPEAKING:
            logger.info("Interruption triggered: user interrupted Friday playback via Speak Now click.")
            # Signal workers to drop stale chunks
            manager.context.interruption_event.set()
            # Stop audio playback IMMEDIATELY — no queue worker latency
            manager.player.interrupt()
            # Handle full interruption lifecycle (state transitions, queue flushes, events)
            import uuid
            active_req_id = manager.context.get_active_request_id() or f"manual-interrupt-{uuid.uuid4()}"
            manager.interruption_manager.handle_interruption(active_req_id, is_manual=True)
            # Clear ptt_active so that the pipeline enters IDLE state without auto-listening
            manager.context.ptt_active.clear()
        else:
            # Clear any stale coordination flags from previous turns
            manager.context.interruption_event.clear()
            manager.context.barge_in_occurred.clear()
            manager.context.ptt_active.set()
            logger.info(f"PTT activated: user clicked Speak Now button in state {current_state.name}.")

    popup.ptt_callback = _ptt_callback

    # Register popup as a listener on pipeline state transitions
    manager.context.register_state_listener(popup.on_state_changed)


    # ── Wire Console Listener ──────────────────────────────────────────────
    listener = ConsolePipelineListener(manager.context, manager.metrics_tracker)
    listener.register()

    # ── Start Pipeline ─────────────────────────────────────────────────────
    print_banner(config)
    manager.start()
    logger.info("Pipeline started. Microphone is active.")

    if sys.modules.get("pytest") is None:
        import uuid
        from core.payloads import SentencePayload
        
        greeting_id = f"greeting-{uuid.uuid4()}"
        manager.context.set_active_request_id(greeting_id)
        manager.context.set_state(PipelineState.SPEAKING)
        
        print("\n[Friday]")
        print("Hello, how can I help you?\n", flush=True)
        
        greeting_payload = SentencePayload(
            request_id=greeting_id,
            text="Hello, how can I help you?",
            is_final=True,
            user_done_timestamp=time.time()
        )
        manager.queue_manager.tts_queue.put(greeting_payload)
    else:
        logger.info("Test environment detected. Skipping interactive greeting.")

    # ── Graceful Shutdown ──────────────────────────────────────────────────
    shutdown_event = manager.context.shutdown_event

    def _signal_handler(sig, frame):
        print("\n\n[Ctrl+C] Shutting down Friday...", flush=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    finally:
        # Print latency report FIRST — nothing blocking before this
        print("\n", flush=True)
        manager.metrics_tracker.print_latency_report()
        from utils.log_utils import report_all
        report_all()

        # Stop workers and destroy popup in a background thread with a hard 4-second deadline
        import threading as _threading
        import os as _os

        def _do_stop():
            try:
                popup.destroy()
                time.sleep(0.5)          # let final audio drain
                manager.stop()
            except Exception as e:
                logger.error(f"Error during pipeline stop: {e}")

        _stop_thread = _threading.Thread(target=_do_stop, daemon=True)
        _stop_thread.start()
        _stop_thread.join(timeout=4.0)   # wait max 4 seconds

        # Force exit regardless — sounddevice C-threads can block join() on Windows
        _os._exit(0)




if __name__ == "__main__":
    main()
