import time
import pytest
from unittest.mock import MagicMock, ANY
from memory.models import Turn, MemorySnapshot, MemoryType
from memory.memory_manager import MemoryManager


def _make_manager(max_recent_turns=5, extra_config=None):
    """Helper to create a MemoryManager with mock dependencies."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "{}"
    mock_metrics = MagicMock()
    config = {
        "system_prompt": "Concisely assist user.",
        "memory": {
            "max_recent_turns": max_recent_turns,
        }
    }
    if extra_config:
        config["memory"].update(extra_config)
    return MemoryManager(config, mock_llm, mock_metrics), mock_llm, mock_metrics


# ──────────────────────────────────────────────
# 1. Core turn flow
# ──────────────────────────────────────────────

def test_memory_manager_turn_and_context():
    """Verify MemoryManager stores messages, extracts facts, builds context and records metrics."""
    manager, mock_llm, mock_metrics = _make_manager()

    manager.add_user_message("What is Artificial Intelligence?")
    assert manager._current_user_query == "What is Artificial Intelligence?"

    # Fact extraction was invoked
    mock_llm.generate.assert_called_once()

    context = manager.get_context("What is Artificial Intelligence?")
    assert "[System Prompt]\nConcisely assist user." in context
    assert "User: What is Artificial Intelligence?" in context

    mock_metrics.record_metric.assert_any_call("context_build_time_ms", ANY)
    mock_metrics.record_metric.assert_any_call("average_context_size", ANY)

    manager.add_assistant_message("AI is machines simulating human intellect.")

    assert manager._current_user_query == ""
    turns = manager.conversation_memory.get_turns()
    assert len(turns) == 1
    assert turns[0].user_message == "What is Artificial Intelligence?"
    assert turns[0].assistant_message == "AI is machines simulating human intellect."
    mock_metrics.record_metric.assert_any_call("memory_turn_count", 1.0)


# ──────────────────────────────────────────────
# 2. Async summarization on window pruning
# ──────────────────────────────────────────────

def test_memory_manager_async_summarization():
    """Verify background summarization fires when max_recent_turns is exceeded."""
    manager, mock_llm, mock_metrics = _make_manager(max_recent_turns=2)
    mock_llm.generate.side_effect = [
        "{}",   # fact T1
        "{}",   # fact T2
        "{}",   # fact T3
        "Merged rolling summary of old turns."
    ]

    manager.add_user_message("T1 user")
    manager.add_assistant_message("T1 assistant")

    manager.add_user_message("T2 user")
    manager.add_assistant_message("T2 assistant")

    assert manager.store.get_summary() == ""

    # Turn 3 causes window to prune Turn 1 → triggers summarization
    manager.add_user_message("T3 user")
    manager.add_assistant_message("T3 assistant")

    time.sleep(0.3)  # allow daemon thread to complete

    assert manager.store.get_summary() == "Merged rolling summary of old turns."
    mock_metrics.record_metric.assert_any_call("summary_count", 1.0)
    mock_metrics.record_metric.assert_any_call("summary_generation_time_ms", ANY)


# ──────────────────────────────────────────────
# 3. get_facts / update_facts
# ──────────────────────────────────────────────

def test_get_facts_returns_empty_initially():
    manager, _, _ = _make_manager()
    assert manager.get_facts() == {}


def test_update_facts_merges_correctly():
    manager, _, _ = _make_manager()
    manager.update_facts({"name": "Supriya"})
    assert manager.get_facts()["name"] == "Supriya"

    manager.update_facts({"preferred_model": "Qwen 2.5"})
    facts = manager.get_facts()
    assert facts["name"] == "Supriya"
    assert facts["preferred_model"] == "Qwen 2.5"


def test_update_facts_overwrites_existing_key():
    manager, _, _ = _make_manager()
    manager.update_facts({"name": "Old"})
    manager.update_facts({"name": "Supriya"})
    assert manager.get_facts()["name"] == "Supriya"


# ──────────────────────────────────────────────
# 4. export_memory / import_memory
# ──────────────────────────────────────────────

def test_export_memory_structure():
    manager, _, _ = _make_manager()
    manager.update_facts({"name": "Supriya"})
    manager.add_user_message("Hello")
    manager.add_assistant_message("Hi there!")

    exported = manager.export_memory()

    assert "recent_turns" in exported
    assert "facts" in exported
    assert "summaries" in exported
    assert "total_turn_count" in exported
    assert exported["facts"]["name"] == "Supriya"
    assert exported["total_turn_count"] == 1
    assert len(exported["recent_turns"]) == 1


def test_import_memory_restores_state():
    manager, _, _ = _make_manager()

    snapshot_data = {
        "recent_turns": [
            {"user_message": "My name is Supriya", "assistant_message": "Nice to meet you!", "timestamp": time.time()}
        ],
        "summaries": ["The user introduced themselves as Supriya."],
        "facts": {"name": "Supriya", "project": "Voice Assistant"},
        "total_turn_count": 1
    }

    manager.import_memory(snapshot_data)

    turns = manager.conversation_memory.get_turns()
    assert len(turns) == 1
    assert turns[0].user_message == "My name is Supriya"

    facts = manager.get_facts()
    assert facts["name"] == "Supriya"
    assert facts["project"] == "Voice Assistant"

    assert manager.store.get_summary() == "The user introduced themselves as Supriya."
    assert manager._total_turn_count == 1


def test_export_import_round_trip():
    manager, _, _ = _make_manager()
    manager.update_facts({"name": "Supriya", "project": "Voice Assistant"})
    manager.add_user_message("Hello")
    manager.add_assistant_message("Hi!")

    exported = manager.export_memory()

    # Create a fresh manager and import
    manager2, _, _ = _make_manager()
    manager2.import_memory(exported)

    assert manager2.get_facts()["name"] == "Supriya"
    assert len(manager2.conversation_memory.get_turns()) == 1
    assert manager2._total_turn_count == 1


# ──────────────────────────────────────────────
# 5. clear_session
# ──────────────────────────────────────────────

def test_clear_session_resets_all_state():
    manager, _, _ = _make_manager()
    manager.update_facts({"name": "Supriya"})
    manager.add_user_message("Hello")
    manager.add_assistant_message("Hi!")

    manager.clear_session()

    assert manager.conversation_memory.get_turns() == []
    assert manager.get_facts() == {}
    assert manager.store.get_summary() == ""
    assert manager._total_turn_count == 0


# ──────────────────────────────────────────────
# 6. MemorySnapshot and MemoryType models
# ──────────────────────────────────────────────

def test_memory_snapshot_to_dict():
    turn = Turn(user_message="Hello", assistant_message="Hi")
    snapshot = MemorySnapshot(
        recent_turns=[turn],
        summaries=["Summary text"],
        facts={"name": "Supriya"},
        total_turn_count=1
    )
    d = snapshot.to_dict()
    assert d["total_turn_count"] == 1
    assert d["facts"]["name"] == "Supriya"
    assert d["summaries"] == ["Summary text"]
    assert len(d["recent_turns"]) == 1


def test_memory_snapshot_from_dict_round_trip():
    turn = Turn(user_message="Hello", assistant_message="Hi")
    snapshot = MemorySnapshot(
        recent_turns=[turn],
        summaries=["A summary"],
        facts={"key": "value"},
        total_turn_count=5
    )
    restored = MemorySnapshot.from_dict(snapshot.to_dict())
    assert restored.total_turn_count == 5
    assert restored.facts["key"] == "value"
    assert len(restored.recent_turns) == 1
    assert restored.summaries == ["A summary"]


def test_memory_type_enum_values():
    assert MemoryType.TURN.value == "turn"
    assert MemoryType.FACT.value == "fact"
    assert MemoryType.SUMMARY.value == "summary"
