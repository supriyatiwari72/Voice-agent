import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FactExtractor:
    """
    Extracts key-value facts from user queries using the active LLM provider.
    """
    def __init__(self, llm: Any):
        self.llm = llm

    def extract_facts(self, user_message: str) -> Dict[str, Any]:
        """
        Queries LLM to find facts in the user message and parse them to a dictionary.
        """
        if not user_message or len(user_message.strip()) < 3:
            return {}

        prompt = (
            "You are a factual information extraction assistant.\n"
            "Analyze the user message and extract key facts about the user's name, profile, preferences, projects, or interests as key-value pairs in JSON format.\n"
            "Only return keys that have been explicitly mentioned or updated. If no new factual details are present, return an empty JSON object {}.\n"
            "Do not include any extra text, comments, explanation, or code block formatting. Return only valid JSON.\n\n"
            f"User Message: \"{user_message}\"\n"
            "JSON output:"
        )

        try:
            logger.info("Triggering LLM for fact extraction...")
            raw_response = self.llm.generate(prompt).strip()
            
            # Extract JSON block using regex
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                facts = json.loads(json_str)
                if isinstance(facts, dict):
                    logger.info(f"Successfully extracted facts: {facts}")
                    return facts
            
            # Try parsing raw response if regex didn't find braces but it starts/ends with them
            facts = json.loads(raw_response)
            if isinstance(facts, dict):
                logger.info(f"Successfully extracted facts directly: {facts}")
                return facts
        except Exception as e:
            logger.warning(f"Failed to extract facts or parse JSON: {e}")
        
        return {}
