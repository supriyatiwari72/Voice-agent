"""
utils/popup.py — Event-Driven, Thread-Safe Status Popup for Friday
=================================================================
A large, clearly readable floating status card with push-to-talk button.

Shows:
  - 🟢 "Speak Now"          [BUTTON] when waiting for user — must click to speak
  - 🔵 "Listening..."       when user is actively speaking
  - 🟡 "Processing..."      when transcribing / thinking
  - 🟣 "Friday is Speaking" when TTS is playing back — speak to interrupt
"""

import logging
import queue
import threading
import time
import tkinter as tk
from typing import Optional, Callable

from pipeline.pipeline_state import PipelineState

logger = logging.getLogger("voice_agent.popup")


class PopupStatus:
    """Enums representing mapped UI states for the popup."""
    LISTENING     = "listening"
    USER_SPEAKING = "user_speaking"
    TRANSCRIBING  = "transcribing"
    THINKING      = "thinking"
    SPEAKING      = "speaking"
    HIDDEN        = "hidden"


class StatusMapper:
    """Decouples internal PipelineState from the UI status representation."""

    @staticmethod
    def map_state(state: PipelineState) -> PopupStatus:
        if state == PipelineState.LISTENING:
            return PopupStatus.LISTENING
        elif state == PipelineState.USER_SPEAKING:
            return PopupStatus.USER_SPEAKING
        elif state == PipelineState.TRANSCRIBING:
            return PopupStatus.TRANSCRIBING
        elif state in (PipelineState.THINKING, PipelineState.GENERATING):
            return PopupStatus.THINKING
        elif state == PipelineState.SPEAKING:
            return PopupStatus.SPEAKING
        elif state == PipelineState.PROCESSING:
            return PopupStatus.TRANSCRIBING
        else:
            return PopupStatus.HIDDEN


# ── Color Palette ────────────────────────────────────────────────────────────
BG          = "#0f1117"   # Very dark background
GREEN       = "#00e676"   # Speak Now
YELLOW      = "#ffd740"   # Thinking / Processing
PURPLE      = "#ce93d8"   # Assistant Speaking
BLUE        = "#82b1ff"   # User Speaking indicator
TEXT_MAIN   = "#ffffff"
TEXT_DIM    = "#9e9e9e"
CARD_BG     = "#1a1d2e"
RED         = "#ff5252"   # Active recording indicator


class StatusPopup:
    """
    Floating Tkinter Status card running in a background thread.
    In LISTENING state shows a clickable 'Speak Now' button (push-to-talk).
    In SPEAKING state user can interrupt by speaking freely (barge-in).
    """

    def __init__(self, debounce_ms: int = 150, ptt_callback: Optional[Callable] = None):
        self.debounce_ms = debounce_ms
        # Called when user clicks the Speak Now button
        self.ptt_callback: Optional[Callable] = ptt_callback

        self.root: Optional[tk.Tk] = None
        self.status_queue: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        self.current_status: str = PopupStatus.HIDDEN
        self.last_update_time: float = 0.0
        self.pending_after_id: Optional[str] = None

        # Animation
        self.crawl_offset = 0
        self.pulse_width = 3.0
        self.pulse_dir = 0.5

        # Drag tracking
        self._drag_x = 0
        self._drag_y = 0

    def start(self) -> None:
        """Starts the Tkinter main loop inside a daemon thread."""
        self.thread = threading.Thread(target=self._run, name="StatusPopupThread", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.root = tk.Tk()
            self._setup_ui()
            self.root.withdraw()
            self._poll_queue()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in Tkinter GUI mainloop thread: {e}", exc_info=True)

    def _setup_ui(self) -> None:
        assert self.root is not None

        W, H = 300, 140

        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        self.root.geometry(f"{W}x{H}")

        # Position: top-right corner
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{W}x{H}+{sw - W - 24}+{30}")

        # Allow dragging
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._drag)

        # ── Outer card frame ─────────────────────────────────────────────────
        self.card = tk.Frame(self.root, bg=CARD_BG, bd=0, relief="flat")
        self.card.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self.card.bind("<Button-1>", self._start_drag)
        self.card.bind("<B1-Motion>", self._drag)

        # Left accent bar (colored indicator)
        self.accent_bar = tk.Frame(self.card, width=5, bg=GREEN)
        self.accent_bar.pack(side=tk.LEFT, fill=tk.Y)

        # Content area
        content = tk.Frame(self.card, bg=CARD_BG)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=14, pady=10)
        content.bind("<Button-1>", self._start_drag)
        content.bind("<B1-Motion>", self._drag)

        # App name / label row
        top_row = tk.Frame(content, bg=CARD_BG)
        top_row.pack(fill=tk.X)

        self.app_label = tk.Label(
            top_row, text="FRIDAY",
            fg=TEXT_DIM, bg=CARD_BG,
            font=("Segoe UI", 8, "bold"),
            anchor="w"
        )
        self.app_label.pack(side=tk.LEFT)

        self.dot_label = tk.Label(
            top_row, text="●",
            fg=GREEN, bg=CARD_BG,
            font=("Segoe UI", 9),
        )
        self.dot_label.pack(side=tk.RIGHT)

        # Main status message
        self.status_label = tk.Label(
            content,
            text="Speak Now",
            fg=GREEN, bg=CARD_BG,
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        )
        self.status_label.pack(fill=tk.X, pady=(2, 0))
        self.status_label.bind("<Button-1>", self._start_drag)
        self.status_label.bind("<B1-Motion>", self._drag)

        # Subtitle / hint
        self.hint_label = tk.Label(
            content,
            text="Click the button below to speak",
            fg=TEXT_DIM, bg=CARD_BG,
            font=("Segoe UI", 9),
            anchor="w"
        )
        self.hint_label.pack(fill=tk.X)
        self.hint_label.bind("<Button-1>", self._start_drag)
        self.hint_label.bind("<B1-Motion>", self._drag)

        # ── Speak Now Button ─────────────────────────────────────────────────
        self.speak_btn = tk.Button(
            content,
            text="🎤  Speak Now",
            fg="#0f1117",
            bg=GREEN,
            activebackground="#00c060",
            activeforeground="#0f1117",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=self._on_speak_clicked
        )
        # Start hidden — will be shown only in LISTENING state
        self.speak_btn.pack_forget()

        # Bottom progress bar canvas
        self.canvas = tk.Canvas(
            self.card, height=4, bg=CARD_BG,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Drag support ──────────────────────────────────────────────────────────
    def _start_drag(self, event: tk.Event) -> None:
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _drag(self, event: tk.Event) -> None:
        assert self.root is not None
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ── Thread-safe state update ──────────────────────────────────────────────
    def on_state_changed(self, state: PipelineState) -> None:
        self.status_queue.put(state)

    def _poll_queue(self) -> None:
        assert self.root is not None
        try:
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
        self.root.after(50, self._poll_queue)

    def _handle_state_update(self, state: PipelineState) -> None:
        assert self.root is not None
        if self.pending_after_id is not None:
            self.root.after_cancel(self.pending_after_id)
            self.pending_after_id = None
        now = time.time() * 1000.0
        elapsed = now - self.last_update_time
        if elapsed >= self.debounce_ms:
            self._apply_state(state)
        else:
            delay = int(self.debounce_ms - elapsed)
            self.pending_after_id = self.root.after(delay, lambda: self._apply_state(state))

    def _apply_state(self, state: PipelineState) -> None:
        assert self.root is not None
        self.last_update_time = time.time() * 1000.0
        self.pending_after_id = None

        status = StatusMapper.map_state(state)
        self.current_status = status

        if status == PopupStatus.HIDDEN:
            self.root.withdraw()
            return

        self.root.deiconify()
        self.root.attributes("-topmost", True)

        # ── SPEAK NOW (LISTENING) — show the clickable button ─────────────────
        if status == PopupStatus.LISTENING:
            self.accent_bar.config(bg=GREEN)
            self.dot_label.config(fg=GREEN)
            self.status_label.config(text="Speak Now", fg=GREEN)
            self.hint_label.config(text="Click the button below to speak")
            self.canvas.delete("bar")
            self.canvas.create_rectangle(0, 0, 300, 4, fill=GREEN, outline="", tags="bar")
            # Re-enable and show the Speak Now button
            self.speak_btn.config(
                text="🎤  Speak Now",
                bg=GREEN,
                fg="#0f1117",
                state="normal"
            )
            self.speak_btn.pack(fill=tk.X, pady=(5, 0))

        # ── USER SPEAKING — hide button, show mic active state ────────────────
        elif status == PopupStatus.USER_SPEAKING:
            self.speak_btn.pack_forget()
            self.accent_bar.config(bg=BLUE)
            self.dot_label.config(fg=BLUE)
            self.status_label.config(text="Listening...", fg=BLUE)
            self.hint_label.config(text="Keep talking, I'm listening.")
            self.canvas.delete("bar")
            self.canvas.create_rectangle(0, 0, 300, 4, fill=BLUE, outline="", tags="bar")

        # ── PROCESSING / TRANSCRIBING ─────────────────────────────────────────
        elif status == PopupStatus.TRANSCRIBING:
            self.speak_btn.pack_forget()
            self.accent_bar.config(bg=YELLOW)
            self.dot_label.config(fg=YELLOW)
            self.status_label.config(text="Processing...", fg=YELLOW)
            self.hint_label.config(text="Converting your speech to text...")
            self.crawl_offset = 0
            self._animate_crawler(YELLOW)

        # ── THINKING ─────────────────────────────────────────────────────────
        elif status == PopupStatus.THINKING:
            self.speak_btn.pack_forget()
            self.accent_bar.config(bg=YELLOW)
            self.dot_label.config(fg=YELLOW)
            self.status_label.config(text="Thinking...", fg=YELLOW)
            self.hint_label.config(text="Friday is generating a response...")
            self.crawl_offset = 0
            self._animate_crawler(YELLOW)

        # ── FRIDAY IS SPEAKING — no button, but barge-in works via VAD ───────
        elif status == PopupStatus.SPEAKING:
            self.speak_btn.pack_forget()
            self.accent_bar.config(bg=PURPLE)
            self.dot_label.config(fg=PURPLE)
            self.status_label.config(text="Friday is Speaking", fg=PURPLE)
            self.hint_label.config(text="Speak anytime to interrupt me.")
            self.pulse_width = 3.0
            self.pulse_dir = 0.5
            self._animate_pulse()

    def _on_speak_clicked(self) -> None:
        """Called when the user clicks the Speak Now button."""
        # Update button to show recording is active
        if self.speak_btn and self.root:
            self.speak_btn.config(
                text="🔴  Listening...",
                bg=RED,
                fg=TEXT_MAIN,
                state="disabled"
            )
        # Fire PTT callback (sets ptt_active event on the pipeline context)
        if self.ptt_callback:
            threading.Thread(target=self.ptt_callback, daemon=True).start()

    # ── Animations ────────────────────────────────────────────────────────────
    def _animate_crawler(self, color: str) -> None:
        if self.root is None or self.current_status not in (
            PopupStatus.TRANSCRIBING, PopupStatus.THINKING
        ):
            return

        self.crawl_offset = (self.crawl_offset + 8) % 300
        self.canvas.delete("bar")

        x1 = self.crawl_offset
        x2 = x1 + 80

        if x2 > 300:
            self.canvas.create_rectangle(0, 0, x2 - 300, 4, fill=color, outline="", tags="bar")
            x2 = 300
        self.canvas.create_rectangle(x1, 0, x2, 4, fill=color, outline="", tags="bar")

        self.root.after(25, lambda: self._animate_crawler(color))

    def _animate_pulse(self) -> None:
        if self.root is None or self.current_status != PopupStatus.SPEAKING:
            return

        self.pulse_width += self.pulse_dir
        if self.pulse_width >= 4.0:
            self.pulse_dir = -0.5
        elif self.pulse_width <= 1.0:
            self.pulse_dir = 0.5

        self.canvas.delete("bar")
        self.canvas.create_rectangle(
            0, 0, 300, int(self.pulse_width * 1.5),
            fill=PURPLE, outline="", tags="bar"
        )
        self.root.after(35, self._animate_pulse)

    # ── Teardown ──────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        if self.root is not None:
            try:
                self.root.after(0, self._destroy_safe)
            except Exception:
                pass

    def _destroy_safe(self) -> None:
        if self.root is not None:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
            self.root = None
