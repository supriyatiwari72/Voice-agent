import pytest
import threading
from memory.models import Turn
from memory.conversation_memory import ConversationMemory

def test_conversation_memory_pruning():
    """
    Verify sliding window prunes oldest turns when size limit is exceeded.
    """
    mem = ConversationMemory(max_turns=3)
    
    # Add 1st turn
    p1 = mem.add_turn(Turn("hello", "hi"))
    assert len(mem.get_turns()) == 1
    assert p1 == []
    
    # Add 2nd and 3rd turns
    mem.add_turn(Turn("how are you?", "doing great"))
    mem.add_turn(Turn("what is AI?", "artificial intelligence"))
    assert len(mem.get_turns()) == 3
    
    # Add 4th turn -> Should prune 1st turn
    p4 = mem.add_turn(Turn("explain transformers", "transformers are..."))
    assert len(mem.get_turns()) == 3
    assert len(p4) == 1
    assert p4[0].user_message == "hello"
    assert p4[0].assistant_message == "hi"
    
    # Verify remaining turns in sliding window
    turns = mem.get_turns()
    assert turns[0].user_message == "how are you?"
    assert turns[1].user_message == "what is AI?"
    assert turns[2].user_message == "explain transformers"

def test_conversation_memory_thread_safety():
    """
    Verify thread-safety under concurrent additions.
    """
    mem = ConversationMemory(max_turns=100)
    num_threads = 10
    turns_per_thread = 20
    
    threads = []
    def worker(tid):
        for j in range(turns_per_thread):
            mem.add_turn(Turn(f"user-{tid}-{j}", f"assistant-{tid}-{j}"))
            
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(mem.get_turns()) == 100 # Pruning should have occurred since limit is 100 and we added 200 total
