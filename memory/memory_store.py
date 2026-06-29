import threading
from typing import Dict, Any, List

class MemoryStore:
    """
    Thread-safe store for rolling summaries, structured key-value facts, and conversation topics.
    """
    def __init__(self):
        self.summary: str = ""
        self.facts: Dict[str, Any] = {}
        self.topics: List[str] = []
        self.lock = threading.Lock()

    def update_summary(self, new_summary: str) -> None:
        with self.lock:
            self.summary = new_summary

    def get_summary(self) -> str:
        with self.lock:
            return self.summary

    def set_facts(self, facts: Dict[str, Any]) -> None:
        with self.lock:
            self.facts = dict(facts)

    def update_facts(self, new_facts: Dict[str, Any]) -> None:
        with self.lock:
            self.facts.update(new_facts)

    def get_facts(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.facts)

    def add_topic(self, topic: str) -> None:
        with self.lock:
            if topic not in self.topics:
                self.topics.append(topic)

    def set_topics(self, topics: List[str]) -> None:
        with self.lock:
            self.topics = list(topics)

    def get_topics(self) -> List[str]:
        with self.lock:
            return list(self.topics)

    def clear(self) -> None:
        with self.lock:
            self.summary = ""
            self.facts.clear()
            self.topics.clear()
