import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any

@dataclass
class Turn:
    user_message: str
    assistant_message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(
            user_message=data.get("user_message", ""),
            assistant_message=data.get("assistant_message", ""),
            timestamp=data.get("timestamp", time.time())
        )

@dataclass
class SessionState:
    turns: List[Turn] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "facts": self.facts,
            "summary": self.summary,
            "topics": self.topics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        turns_data = data.get("turns", [])
        turns = [Turn.from_dict(t) for t in turns_data]
        return cls(
            turns=turns,
            facts=data.get("facts", {}),
            summary=data.get("summary", ""),
            topics=data.get("topics", [])
        )

class MemoryType(Enum):
    """Enumerates the kinds of data stored in the memory layer."""
    TURN = "turn"
    FACT = "fact"
    SUMMARY = "summary"

@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of the full memory state — used for export/import."""
    recent_turns: List["Turn"] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    total_turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_turns": [t.to_dict() for t in self.recent_turns],
            "summaries": self.summaries,
            "facts": self.facts,
            "total_turn_count": self.total_turn_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySnapshot":
        turns = [Turn.from_dict(t) for t in data.get("recent_turns", [])]
        return cls(
            recent_turns=turns,
            summaries=data.get("summaries", []),
            facts=data.get("facts", {}),
            total_turn_count=data.get("total_turn_count", 0)
        )
