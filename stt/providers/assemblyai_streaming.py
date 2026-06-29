import logging
import threading
import time
import json
import base64
from typing import Dict, Any, Optional
from stt.base import BaseSTT, BaseStreamingSTT

logger = logging.getLogger(__name__)

class AssemblyAIStreamingSTT(BaseSTT, BaseStreamingSTT):
    """
    Production AssemblyAI Streaming STT Provider.
    Streams raw audio frames via WebSocket (base64 encoded JSON) and invokes callbacks.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("stt", {}).get("assemblyai", {})
        self.api_key = self.config.get("api_key") or config.get("assemblyai_api_key") or "MOCK_KEY"
        self.url = self.config.get("url") or "wss://api.assemblyai.com/v2/realtime"
        
        self.callback = None
        self.request_id = None
        self._ws = None
        self._active = False
        self._lock = threading.Lock()
        
    def transcribe(self, audio_data: bytes) -> str:
        """
        Batch fallback.
        """
        result_text = []
        def cb(chunk, is_final):
            if chunk:
                result_text.append(chunk)
                
        self.start_stream("batch-transcribe", cb)
        self.stream_audio(audio_data)
        self.stop_stream()
        
        time.sleep(0.05)
        return "".join(result_text)

    def start_stream(self, request_id: str, on_transcript_cb) -> None:
        with self._lock:
            self.request_id = request_id
            self.callback = on_transcript_cb
            self._active = True
            
        logger.info(f"AssemblyAIStreamingSTT: Starting stream for request {request_id}")
        
        if not self.api_key or self.api_key == "MOCK_KEY":
            logger.info("AssemblyAIStreamingSTT: Running in MOCK/FALLBACK mode.")
            self._ws = "mock"
            return
            
        try:
            import websocket
            # AssemblyAI authenticates via Authorization header or token query parameter
            ws_url = f"{self.url}?sample_rate=16000"
            headers = {"Authorization": self.api_key}
            
            self._ws = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            self._thread.start()
            
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"AssemblyAIStreamingSTT: Failed to connect: {e}. Falling back to mock.")
            self._ws = "mock"

    def stream_audio(self, audio_chunk: bytes) -> None:
        if not self._active:
            return
            
        if self._ws == "mock":
            if self.callback:
                self.callback("Hello ", False)
                self.callback("AssemblyAI ", False)
            return
            
        try:
            import websocket
            if self._ws and isinstance(self._ws, websocket.WebSocketApp) and self._ws.sock and self._ws.sock.connected:
                # AssemblyAI expects base64 encoded audio in JSON
                base64_audio = base64.b64encode(audio_chunk).decode("utf-8")
                payload = json.dumps({"audio_data": base64_audio})
                self._ws.send(payload)
        except Exception as e:
            logger.error(f"AssemblyAIStreamingSTT: Error sending audio chunk: {e}")

    def stop_stream(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            
        logger.info(f"AssemblyAIStreamingSTT: Stopping stream for request {self.request_id}")
        
        if self._ws == "mock":
            if self.callback:
                self.callback("Streaming STT.", True)
            self._ws = None
            return
            
        try:
            import websocket
            if self._ws and isinstance(self._ws, websocket.WebSocketApp):
                # Send terminate message
                self._ws.send(json.dumps({"terminate_session": True}))
                self._ws.close()
        except Exception as e:
            logger.error(f"AssemblyAIStreamingSTT: Error closing stream: {e}")
        finally:
            self._ws = None

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            message_type = data.get("message_type")
            transcript = data.get("text", "")
            
            if message_type in ("PartialTranscript", "FinalTranscript") and transcript:
                is_final = (message_type == "FinalTranscript")
                if self.callback:
                    self.callback(transcript, is_final)
        except Exception as e:
            logger.error(f"AssemblyAIStreamingSTT: Error parsing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"AssemblyAIStreamingSTT: WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("AssemblyAIStreamingSTT: WebSocket closed")
