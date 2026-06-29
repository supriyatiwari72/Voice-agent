import threading
from typing import List
from memory.models import Turn

class ConversationMemory:
    """
    Thread-safe sliding window of recent conversation turns.
    """
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[Turn] = []
        self.lock = threading.Lock()

    def add_turn(self, turn: Turn) -> List[Turn]:
        """
        Adds a turn to the sliding window. Prunes older turns if size exceeds max_turns.
        Returns the list of pruned turns.
        """
        pruned = []
        with self.lock:
            self.turns.append(turn)
            while len(self.turns) > self.max_turns:
                pruned.append(self.turns.pop(0))
        return pruned

    def get_turns(self) -> List[Turn]:
        """
        Returns a copy of the current turns in memory.
        """
        with self.lock:
            return list(self.turns)

    def set_turns(self, turns: List[Turn]) -> None:
        """
        Sets the sliding window turns directly (e.g. on restore).
        """
        with self.lock:
            self.turns = list(turns)

    def clear(self) -> None:
        """
        Clears the turns sliding window.
        """
        with self.lock:
            self.turns.clear()
