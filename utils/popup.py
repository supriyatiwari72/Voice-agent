"""
utils/popup.py — Event-Driven, Thread-Safe Status Popup for Friday
=================================================================
A floating, borderless, draggable status card utilizing a Tokyo Night
color palette. Observes pipeline state transitions thread-safely via a Queue
and updates widgets/animations exclusively on the Tkinter main thread.
"""

import logging
import queue
import threading
import time
import tkinter as tk
from typing import Optional

from pipeline.pipeline_state import PipelineState

logger = logging.getLogger("voice_agent.popup")


class PopupStatus:
    """Enums representing mapped UI states for the popup."""
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HIDDEN = "hidden"


class StatusMapper:
    """Decouples internal PipelineState from the UI status representation."""

    @staticmethod
    def map_state(state: PipelineState) -> PopupStatus:
        if state == PipelineState.LISTENING:
            return PopupStatus.LISTENING
        elif state == PipelineState.TRANSCRIBING:
            return PopupStatus.TRANSCRIBING
        elif state == PipelineState.THINKING:
            return PopupStatus.THINKING
        elif state == PipelineState.SPEAKING:
            return PopupStatus.SPEAKING
        elif state in (PipelineState.IDLE, PipelineState.PROCESSING):
            # Processing states act as progression indicators
            return PopupStatus.TRANSCRIBING
        else:
            return PopupStatus.HIDDEN


class StatusPopup:
    """
    Floating Tkinter Status card running in a background thread.
    Updates are thread-safe and animated natively on the Tkinter event loop.
    """

    def __init__(self, debounce_ms: int = 200):
        self.debounce_ms = debounce_ms
        self.root: Optional[tk.Tk] = None
        self.status_queue: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        # UI state variables (accessed only from Tkinter thread)
        self.current_status: PopupStatus = PopupStatus.HIDDEN
        self.last_update_time: float = 0.0
        self.pending_after_id: Optional[str] = None

        # Animation states
        self.crawl_offset = 0
        self.pulse_width = 3.0
        self.pulse_dir = 0.4

        # Drag tracking
        self._drag_x = 0
        self._drag_y = 0

    def start(self) -> None:
        """Starts the Tkinter main loop inside a daemon thread."""
        self.thread = threading.Thread(target=self._run, name="StatusPopupThread", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        """Initializes and displays the Tkinter GUI."""
        try:
            self.root = tk.Tk()
            self._setup_ui()
            # Set initial state to hidden
            self.root.withdraw()
            
            # Start queue polling loop
            self._poll_queue()
            
            # Start Tkinter main loop
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in Tkinter GUI mainloop thread: {e}", exc_info=True)

    def _setup_ui(self) -> None:
        """Configures the Tokyo Night styled, borderless widget layout."""
        assert self.root is not None
        
        # Borderless, floating window always on top
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1b26")  # Tokyo Night background
        self.root.geometry("260x85")

        # Center card on the screen
        self.root.update_idletasks()
        w, h = 260, 85
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # Position at top right or center top
        x = sw - w - 30
        y = 50
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Setup mouse dragging support
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._drag)

        # Padding frame
        self.frame = tk.Frame(self.root, bg="#1a1b26")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Title Label
        self.icon_label = tk.Label(
            self.frame,
            text="🎤 Friday",
            fg="#c0caf5",
            bg="#1a1b26",
            font=("Segoe UI", 11, "bold"),
            anchor="w"
        )
        self.icon_label.pack(fill=tk.X, anchor="w")

        # Status subtitle text
        self.status_label = tk.Label(
            self.frame,
            text="Speak now...",
            fg="#00FF88",
            bg="#1a1b26",
            font=("Segoe UI", 12),
            anchor="w"
        )
        self.status_label.pack(fill=tk.X, anchor="w", pady=(2, 6))

        # Bottom indicator line canvas
        self.canvas = tk.Canvas(
            self.frame,
            height=6,
            bg="#1a1b26",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.X, anchor="s")

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_x = event.x
        self._drag_y = event.y

    def _drag(self, event: tk.Event) -> None:
        assert self.root is not None
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def on_state_changed(self, state: PipelineState) -> None:
        """
        Thread-safe observer callback registered on the PipelineContext.
        Pushes state transitions to the Tkinter update queue.
        """
        self.status_queue.put(state)

    def _poll_queue(self) -> None:
        """Polls the thread-safe queue and schedules state updates with debouncing."""
        assert self.root is not None
        try:
            # Process all pending updates in queue, keeping only the latest state
            latest_state = None
            while True:
                try:
                    latest_state = self.status_queue.get_nowait()
                    self.status_queue.task_done()
                except queue.Empty:
                    break

            if latest_state is not None:
                self._handle_state_update(latest_state)

        except Exception as e:
            logger.error(f"Error during popup queue polling: {e}")
        
        # Schedule the next queue check in 50ms
        self.root.after(50, self._poll_queue)

    def _handle_state_update(self, state: PipelineState) -> None:
        """Debounces and schedules the visual state transition."""
        assert self.root is not None
        
        # Cancel any scheduled pending updates
        if self.pending_after_id is not None:
            self.root.after_cancel(self.pending_after_id)
            self.pending_after_id = None

        now = time.time() * 1000.0
        elapsed = now - self.last_update_time

        # If debounce window has passed, apply state immediately.
        # Otherwise, delay update until debounce interval is met.
        if elapsed >= self.debounce_ms:
            self._apply_state(state)
        else:
            delay = int(self.debounce_ms - elapsed)
            self.pending_after_id = self.root.after(delay, lambda: self._apply_state(state))

    def _apply_state(self, state: PipelineState) -> None:
        """Translates PipelineState and updates window visibility, widgets, and animation loops."""
        assert self.root is not None
        self.last_update_time = time.time() * 1000.0
        self.pending_after_id = None

        # Decoupled Mapping Layer
        status = StatusMapper.map_state(state)
        self.current_status = status

        if status == PopupStatus.HIDDEN:
            self.root.withdraw()
            return

        # Show window if hidden
        self.root.deiconify()
        self.root.attributes("-topmost", True)

        # Style updates based on the active status
        if status == PopupStatus.LISTENING:
            self.icon_label.config(text="🎤 Friday")
            self.status_label.config(text="Speak now...", fg="#00FF88")
            self.canvas.delete("animation")
            self.canvas.create_line(0, 3, 260, 3, fill="#00FF88", width=3, tags="animation")

        elif status == PopupStatus.TRANSCRIBING:
            self.icon_label.config(text="⚙ Friday")
            self.status_label.config(text="Transcribing...", fg="#FFB300")
            self.crawl_offset = 0
            self._animate_crawler()

        elif status == PopupStatus.THINKING:
            self.icon_label.config(text="🤖 Friday")
            self.status_label.config(text="Generating Response...", fg="#FFB300")
            self.crawl_offset = 0
            self._animate_crawler()

        elif status == PopupStatus.SPEAKING:
            self.icon_label.config(text="🔊 Friday")
            self.status_label.config(text="Answering...", fg="#CC66FF")
            self.pulse_width = 3.0
            self.pulse_dir = 0.4
            self._animate_pulse()

    def _animate_crawler(self) -> None:
        """Natively drives the crawling orange progression bar using the Tkinter event loop."""
        if self.root is None or self.current_status not in (PopupStatus.TRANSCRIBING, PopupStatus.THINKING):
            return

        self.crawl_offset = (self.crawl_offset + 6) % 260
        self.canvas.delete("animation")

        x1 = self.crawl_offset
        x2 = x1 + 60
        
        # Render wrap-around line segment
        if x2 > 260:
            self.canvas.create_line(0, 3, x2 - 260, 3, fill="#FFB300", width=3, tags="animation")
            x2 = 260

        self.canvas.create_line(x1, 3, x2, 3, fill="#FFB300", width=3, tags="animation")
        
        # Frame schedule at ~30 FPS
        self.root.after(30, self._animate_crawler)

    def _animate_pulse(self) -> None:
        """Natively drives the pulsing magenta/purple speaks indicator using the Tkinter event loop."""
        if self.root is None or self.current_status != PopupStatus.SPEAKING:
            return

        self.pulse_width += self.pulse_dir
        if self.pulse_width >= 6.0:
            self.pulse_dir = -0.4
        elif self.pulse_width <= 2.0:
            self.pulse_dir = 0.4

        self.canvas.delete("animation")
        self.canvas.create_line(0, 3, 260, 3, fill="#CC66FF", width=int(self.pulse_width), tags="animation")

        # Frame schedule at ~25 FPS
        self.root.after(40, self._animate_pulse)

    def destroy(self) -> None:
        """Safely tears down the Tkinter window and halts event loops."""
        if self.root is not None:
            # We must destroy the Tkinter window thread-safely
            try:
                self.root.after(0, self._destroy_safe)
            except Exception:
                pass

    def _destroy_safe(self) -> None:
        """Internal helper to execute destroy inside the Tkinter loop."""
        if self.root is not None:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
            self.root = None
