import os
import pytest
from memory.models import SessionState, Turn
from memory.persistence import MemoryPersistence

def test_persistence_save_and_load(tmp_path):
    """
    Verify full SessionState serialization and restoration from disk.
    """
    state = SessionState(
        turns=[Turn("hello", "hi"), Turn("what is my name?", "Supriya")],
        facts={"name": "Supriya", "project": "Voice Agent"},
        summary="User discussed name and project.",
        topics=["Introduction", "Project overview"]
    )
    
    file_path = os.path.join(tmp_path, "session.json")
    
    # Save session
    success = MemoryPersistence.save_session(state, file_path)
    assert success is True
    assert os.path.exists(file_path) is True
    
    # Load session
    loaded_state = MemoryPersistence.load_session(file_path)
    
    assert len(loaded_state.turns) == 2
    assert loaded_state.turns[0].user_message == "hello"
    assert loaded_state.turns[1].assistant_message == "Supriya"
    assert loaded_state.facts == {"name": "Supriya", "project": "Voice Agent"}
    assert loaded_state.summary == "User discussed name and project."
    assert loaded_state.topics == ["Introduction", "Project overview"]

def test_persistence_missing_file():
    """
    Verify loading missing files returns clean/empty SessionState instead of failing.
    """
    state = MemoryPersistence.load_session("non_existent_file.json")
    assert isinstance(state, SessionState)
    assert len(state.turns) == 0
    assert state.facts == {}
    assert state.summary == ""
    assert state.topics == []
