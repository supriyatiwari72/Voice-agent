import json
import logging
import time
import threading
from typing import List, Dict, Any, Optional

from memory.models import Turn, SessionState, MemorySnapshot
from memory.conversation_memory import ConversationMemory
from memory.memory_store import MemoryStore
from memory.summarizer import MemorySummarizer
from memory.fact_extractor import FactExtractor
from memory.persistence import MemoryPersistence
from memory.context_builder import ContextBuilder
from memory.memory_metrics import MemoryMetrics

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Central orchestrator coordinating the short-term window, structured facts,
    rolling summaries, session persistence, and telemetry metrics.

    This is the ONLY public interface through which all workers interact with memory.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, metrics_tracker: Any):
        self.config = config or {}
        self.llm = llm

        # Load all configurable limits from memory section
        memory_config = self.config.get("memory", {})
        max_recent_turns       = memory_config.get("max_recent_turns", 20)
        self._max_turns_before_summary = memory_config.get("max_turns_before_summary", 50)
        self._max_context_chars        = memory_config.get("max_context_chars", 12000)

        # Instantiate sub-modules
        self.conversation_memory = ConversationMemory(max_turns=max_recent_turns)
        self.store = MemoryStore()
        self.summarizer = MemorySummarizer(llm)
        self.fact_extractor = FactExtractor(llm)
        self.metrics = MemoryMetrics(metrics_tracker)

        self._current_user_query: str = ""
        self._total_turn_count: int = 0
        self._summary_count: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core memory workflow API
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """
        Stores user input and triggers synchronous fact extraction so facts are
        available for context building before the LLM call.
        """
        with self._lock:
            self._current_user_query = text

        try:
            extracted_facts = self.fact_extractor.extract_facts(text)
            if extracted_facts:
                self.store.update_facts(extracted_facts)
        except Exception as e:
            logger.error(f"Error extracting facts in MemoryManager: {e}")

    def add_assistant_message(self, text: str) -> None:
        """
        Records the assistant reply as a completed Turn, and triggers
        asynchronous background summarization if thresholds are exceeded.
        """
        user_query = ""
        with self._lock:
            user_query = self._current_user_query
            self._current_user_query = ""
            self._total_turn_count += 1

        if not user_query:
            logger.warning("add_assistant_message called but no user query was tracked. Using placeholder.")
            user_query = "..."

        turn = Turn(user_message=user_query, assistant_message=text)

        # Add to sliding turn memory window; get back any pruned turns
        pruned_turns = self.conversation_memory.add_turn(turn)

        # Record turn count metric
        current_turns_count = len(self.conversation_memory.get_turns())
        self.metrics.record_turn_count(current_turns_count)

        # Determine if any summarization trigger is satisfied
        self._maybe_trigger_summarization(pruned_turns)

    def get_context(self, user_query: str) -> str:
        """
        Assembles and formats the memory-enriched system prompt for the LLM.
        """
        start_time = time.time()

        summary   = self.store.get_summary()
        facts     = self.store.get_facts()
        recent_turns = self.conversation_memory.get_turns()
        system_prompt = self.config.get("system_prompt", "")

        context = ContextBuilder.build_context(
            system_prompt=system_prompt,
            summary=summary,
            facts=facts,
            recent_turns=recent_turns,
            user_query=user_query
        )

        build_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_context_build_time(build_time_ms)
        self.metrics.record_context_size(len(context))

        return context

    def summarize_if_needed(self, pruned_turns: List[Turn] = None) -> None:
        """
        External hook — callers can request background summarization directly.
        Spawns a daemon thread; never blocks the voice pipeline.
        """
        if not pruned_turns:
            return
        self._spawn_summarization_thread(pruned_turns)

    def clear_session(self) -> None:
        """Clears all in-memory structures."""
        self.conversation_memory.clear()
        self.store.clear()
        with self._lock:
            self._summary_count = 0
            self._total_turn_count = 0
            self._current_user_query = ""

    # ------------------------------------------------------------------
    # Fact management
    # ------------------------------------------------------------------

    def get_facts(self) -> Dict[str, Any]:
        """Returns all currently known user facts."""
        return self.store.get_facts()

    def update_facts(self, facts: Dict[str, Any]) -> None:
        """Merges new facts into the memory store."""
        self.store.update_facts(facts)

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_memory(self) -> Dict[str, Any]:
        """
        Exports a full MemorySnapshot dict — suitable for logging, debugging,
        or passing to a future RAG indexing pipeline.
        """
        snapshot = MemorySnapshot(
            recent_turns=self.conversation_memory.get_turns(),
            summaries=[self.store.get_summary()] if self.store.get_summary() else [],
            facts=self.store.get_facts(),
            total_turn_count=self._total_turn_count
        )
        return snapshot.to_dict()

    def import_memory(self, data: Dict[str, Any]) -> None:
        """
        Restores memory state from a MemorySnapshot dict (e.g. exported earlier).
        """
        snapshot = MemorySnapshot.from_dict(data)
        self.conversation_memory.set_turns(snapshot.recent_turns)
        self.store.set_facts(snapshot.facts)
        if snapshot.summaries:
            self.store.update_summary(snapshot.summaries[-1])
        with self._lock:
            self._total_turn_count = snapshot.total_turn_count
            self._summary_count = len(snapshot.summaries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_session(self, file_path: str) -> bool:
        """Saves current memory state to a JSON file."""
        state = SessionState(
            turns=self.conversation_memory.get_turns(),
            facts=self.store.get_facts(),
            summary=self.store.get_summary(),
            topics=self.store.get_topics()
        )
        return MemoryPersistence.save_session(state, file_path)

    def load_session(self, file_path: str) -> bool:
        """Loads memory state from a JSON file and restores it."""
        state = MemoryPersistence.load_session(file_path)
        self.conversation_memory.set_turns(state.turns)
        self.store.set_facts(state.facts)
        self.store.update_summary(state.summary)
        self.store.set_topics(state.topics)
        with self._lock:
            self._total_turn_count = len(state.turns)
            self._summary_count = 1 if state.summary else 0
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_trigger_summarization(self, pruned_turns: List[Turn]) -> None:
        """
        Checks whether any summarization threshold is exceeded and if so spawns
        a background summarization thread.

        Triggers when:
          - Turns were pruned from the sliding window (window overflow), OR
          - Total turn count has crossed max_turns_before_summary, OR
          - The current context string exceeds max_context_chars
        """
        should_summarize = False
        turns_for_summary: List[Turn] = list(pruned_turns) if pruned_turns else []

        with self._lock:
            total = self._total_turn_count

        # Trigger 1 — window pruned old turns
        if pruned_turns:
            should_summarize = True

        # Trigger 2 — total turn count threshold
        if total > 0 and total % self._max_turns_before_summary == 0:
            should_summarize = True
            if not turns_for_summary:
                turns_for_summary = self.conversation_memory.get_turns()

        # Trigger 3 — context character length threshold
        if not should_summarize:
            context_len = len(self.store.get_summary()) + sum(
                len(t.user_message) + len(t.assistant_message)
                for t in self.conversation_memory.get_turns()
            )
            if context_len > self._max_context_chars:
                should_summarize = True
                if not turns_for_summary:
                    turns_for_summary = self.conversation_memory.get_turns()

        if should_summarize and turns_for_summary:
            self._spawn_summarization_thread(turns_for_summary)

    def _spawn_summarization_thread(self, turns: List[Turn]) -> None:
        """Spawns a daemon background thread for async summarization."""
        thread = threading.Thread(
            target=self._run_async_summarization,
            args=(turns,),
            daemon=True
        )
        thread.start()

    def _run_async_summarization(self, pruned_turns: List[Turn]) -> None:
        """Background task executing local LLM summarization — never blocks the pipeline."""
        start_time = time.time()
        try:
            existing_summary = self.store.get_summary()
            new_summary = self.summarizer.generate_summary(existing_summary, pruned_turns)
            self.store.update_summary(new_summary)

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_summary_generation_time(duration_ms)

            with self._lock:
                self._summary_count += 1
                current_summary_count = self._summary_count
            self.metrics.record_summary_count(current_summary_count)

        except Exception as e:
            logger.error(f"Error during async summarization in background: {e}")
