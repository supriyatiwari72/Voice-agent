import json
import logging
import os
import threading
import time
import websocket
from typing import Dict, Any, Optional
from stt.base import BaseSTT, BaseStreamingSTT

logger = logging.getLogger(__name__)

class DeepgramStreamingSTT(BaseSTT, BaseStreamingSTT):
    """
    Production Deepgram Streaming STT Provider.
    Uses websocket-client with proper connection tracking to avoid
    sending audio before the socket is ready.
    """

    def __init__(self, config: Dict[str, Any]):
        models_meta = config.get("models_meta", {})
        stt_config = (
            models_meta.get("stt_providers", {}).get("deepgram_streaming", {})
            or models_meta.get("stt_providers", {}).get("deepgram", {})
            or config.get("stt", {}).get("deepgram", {})
            or config.get("deepgram_streaming", {})
            or {}
        )

        self.api_key = (
            stt_config.get("api_key")
            or config.get("deepgram_api_key")
            or os.environ.get("DEEPGRAM_API_KEY", "")
        )
        self.url = stt_config.get("url") or "wss://api.deepgram.com/v1/listen"
        self.model = stt_config.get("model", "nova-2")
        self.language = stt_config.get("language", "en")
        self.smart_format = stt_config.get("smart_format", True)

        self.callback = None
        self.request_id = None
        self._ws = None
        self._active = False
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._ws_thread = None

        is_mock_key = not self.api_key or self.api_key in ("MOCK_KEY", "mock", "test")
        self.fallback = is_mock_key

        if self.fallback:
            logger.warning("DeepgramStreamingSTT: No valid DEEPGRAM_API_KEY found. Falling back to mock.")

    def _on_open(self, ws):
        self._connected.set()
        logger.info("DeepgramStreamingSTT: WebSocket opened successfully")

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected.clear()
        logger.info(f"DeepgramStreamingSTT: WebSocket closed: {close_status_code} {close_msg}")

    def transcribe(self, audio_data: bytes) -> str:
        if self.fallback:
            return "Deepgram Streaming STT mock result."
        try:
            import websocket
            headers = {"Authorization": f"Token {self.api_key}"}
            ws_url = (
                f"{self.url}?encoding=linear16&sample_rate=16000&channels=1"
                f"&model={self.model}&language={self.language}&smart_format={str(self.smart_format).lower()}"
            )

            result_text = []

            def cb(chunk, is_final):
                if chunk:
                    result_text.append(chunk)

            self.start_stream("batch-transcribe", cb)
            self.stream_audio(audio_data)
            self.stop_stream()
            time.sleep(0.1)
            return "".join(result_text)
        except Exception as e:
            logger.error(f"DeepgramStreamingSTT batch transcription error: {e}")
            return ""

    def start_stream(self, request_id: str, on_transcript_cb) -> None:
        with self._lock:
            self.request_id = request_id
            self.callback = on_transcript_cb
            self._active = True
            self._connected.clear()

        logger.info(f"DeepgramStreamingSTT: Starting stream for request {request_id}")

        if self.fallback:
            logger.info("DeepgramStreamingSTT: Running in MOCK/FALLBACK mode.")
            self._ws = "mock"
            self._connected.set()
            return

        try:
            import websocket
            headers = {"Authorization": f"Token {self.api_key}"}
            ws_url = (
                f"{self.url}?encoding=linear16&sample_rate=16000&channels=1"
                f"&model={self.model}&language={self.language}&smart_format={str(self.smart_format).lower()}"
            )

            for attempt in range(3):
                try:
                    self._ws = websocket.WebSocketApp(
                        ws_url,
                        header=headers,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                    )

                    self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
                    self._ws_thread.start()

                    if self._connected.wait(timeout=15.0):
                        logger.info(f"DeepgramStreamingSTT: Connected on attempt {attempt + 1}.")
                        return
                    else:
                        logger.warning(f"DeepgramStreamingSTT: Connection attempt {attempt + 1} timed out.")
                        if self._ws:
                            try:
                                self._ws.close()
                            except Exception:
                                pass
                        self._ws = None
                except Exception as e:
                    logger.warning(f"DeepgramStreamingSTT: Connection attempt {attempt + 1} failed: {e}")

            logger.error(
                "DeepgramStreamingSTT: All 3 connection attempts failed. "
                "Check your network, DNS resolution, or DEEPGRAM_API_KEY. Falling back to mock."
            )
            self.fallback = True
            self._ws = "mock"
            self._connected.set()
        except Exception as e:
            logger.error(f"DeepgramStreamingSTT: Failed to start stream: {e}. Falling back to mock.")
            self.fallback = True
            self._ws = "mock"
            self._connected.set()
        except Exception as e:
            logger.error(f"DeepgramStreamingSTT: Failed to start stream: {e}. Falling back to mock.")
            self.fallback = True
            self._ws = "mock"
            self._connected.set()

    def stream_audio(self, audio_chunk: bytes) -> None:
        if not self._active:
            return

        if self._ws == "mock":
            if self.callback:
                self.callback("Deepgram streaming result. ", False)
            return

        # Wait for the socket to be fully connected before sending
        if not self._connected.is_set():
            if not self._connected.wait(timeout=3.0):
                logger.warning("DeepgramStreamingSTT: Not connected, dropping audio chunk.")
                return

        try:
            if self._ws and isinstance(self._ws, websocket.WebSocketApp):
                self._ws.send(audio_chunk, opcode=0x2)
        except Exception as e:
            err = str(e).lower()
            if "already closed" not in err and "connection is already closed" not in err:
                logger.error(f"DeepgramStreamingSTT: Error sending audio chunk: {e}")

    def stop_stream(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False

        logger.info(f"DeepgramStreamingSTT: Stopping stream for request {self.request_id}")

        if self._ws == "mock":
            if self.callback:
                self.callback("Streaming STT.", True)
            self._connected.clear()
            self._ws = None
            return

        try:
            import websocket
            if self._ws and isinstance(self._ws, websocket.WebSocketApp):
                try:
                    self._ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
                self._ws.close()
        except Exception as e:
            err = str(e).lower()
            if "already closed" not in err:
                logger.error(f"DeepgramStreamingSTT: Error closing stream: {e}")
        finally:
            self._connected.clear()
            self._ws = None

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
            is_final = data.get("is_final", False)
            if transcript and self.callback:
                self.callback(transcript, is_final)
        except Exception as e:
            logger.error(f"DeepgramStreamingSTT: Error parsing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"DeepgramStreamingSTT: WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected.clear()
        logger.info(f"DeepgramStreamingSTT: WebSocket closed: {close_status_code} {close_msg}")

    def _on_open(self, ws):
        self._connected.set()
        logger.info("DeepgramStreamingSTT: WebSocket opened successfully")
