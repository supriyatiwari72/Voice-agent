import json
import logging
import os
import re
from typing import Dict, Any, List

from knowledge.base import BaseKnowledgeRetriever

logger = logging.getLogger(__name__)


class JSONKnowledgeRetriever(BaseKnowledgeRetriever):
    """
    Concrete knowledge retriever that loads entries from a local JSON file
    and performs case-insensitive keyword matching across searchable fields.
    """

    # Fields searched during retrieval
    SEARCHABLE_FIELDS = [
        "title",
        "question",
        "answer",
        "keywords",
        "related_topics",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the JSON knowledge retriever by loading the knowledge file.

        Args:
            config (Dict[str, Any]): Configuration settings map.
        """
        self.config = config or {}
        self.top_k = self.config.get("top_k", 5)

        application_config = self.config.get("application", {})
        app_name = application_config.get("active", "default")

        # Resolve knowledge file path
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.knowledge_file = os.path.join(
            package_dir,
            "data",
            app_name,
            "knowledge.json",
        )

        self.entries: List[Dict[str, Any]] = []
        self.platform: Dict[str, Any] = {}

        self._load_knowledge()

    def _load_knowledge(self) -> None:
        """
        Loads the knowledge JSON file into memory once during initialization.
        Supports:
        - List format
        - {"entries": [...]}
        - {"platform": {...}, "knowledge": [...]}
        """
        if not os.path.exists(self.knowledge_file):
            logger.warning(f"Knowledge file not found: {self.knowledge_file}")
            return

        try:
            with open(self.knowledge_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                with open(self.knowledge_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                print("\n================ KNOWLEDGE DEBUG ================")
                print("Knowledge file:", self.knowledge_file)
                print("Python type:", type(data))

                if isinstance(data, dict):
                    print("Keys:", list(data.keys()))
                    print("Has platform:", "platform" in data)
                    print("Has knowledge:", "knowledge" in data)
                    print("Has entries:", "entries" in data)

                elif isinstance(data, list):
                    print("JSON is a list")
                    print("Length:", len(data))

                print("=================================================\n")

            if isinstance(data, list):
                # Legacy format
                self.entries = data

            elif isinstance(data, dict):

                # Store platform metadata if available
                self.platform = data.get("platform", {})

                if "knowledge" in data:
                    self.entries = data["knowledge"]

                elif "entries" in data:
                    self.entries = data["entries"]

                else:
                    logger.warning(
                        f"Unexpected knowledge file format in: {self.knowledge_file}"
                    )
                    return

            else:
                logger.warning(
                    f"Unexpected knowledge file format in: {self.knowledge_file}"
                )
                return

            logger.info(
                f"Loaded {len(self.entries)} knowledge entries from: {self.knowledge_file}"
            )

            if self.platform:
                logger.info(
                    f"Platform: "
                    f"{self.platform.get('name', 'Unknown')} "
                    f"({self.platform.get('domain', 'Unknown')})"
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse knowledge JSON file: {e}")

        except Exception as e:
            logger.exception(f"Failed to load knowledge file: {e}")

    def _compute_relevance(
        self,
        entry: Dict[str, Any],
        query_tokens: List[str],
    ) -> int:
        """
        Computes a simple relevance score for a knowledge entry.
        """

        score = 0

        for field in self.SEARCHABLE_FIELDS:

            value = entry.get(field)

            if value is None:
                continue

            if isinstance(value, list):
                field_text = re.sub(
                    r"[^\w\s]",
                    " ",
                    " ".join(str(item) for item in value).lower(),
                )
            else:
                field_text = re.sub(
                    r"[^\w\s]",
                    " ",
                    str(value).lower(),
                )

            for token in query_tokens:
                if token in field_text:
                    score += 1

        return score

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Returns the top-k most relevant knowledge entries.
        """

        if not query or not query.strip():
            return []

        if not self.entries:
            return []

        query_tokens = re.findall(r"\w+", query.lower())

        scored_entries = []

        for entry in self.entries:
            score = self._compute_relevance(entry, query_tokens)

            if score > 0:
                scored_entries.append((score, entry))

        scored_entries.sort(
            key=lambda x: x[0],
            reverse=True,
        )
        
        print("\n========== RETRIEVAL DEBUG ==========")
        print("Query:", query)
        print("Tokens:", query_tokens)
        print("Entries Loaded:", len(self.entries))
        print("Matches Found:", len(scored_entries))

        for score, entry in scored_entries[:5]:
            print(score, "->", entry.get("title"))

        print("====================================\n")

        return [
            entry
            for _, entry in scored_entries[: self.top_k]
        ]