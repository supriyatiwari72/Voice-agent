import logging
from typing import List, Any
from memory.models import Turn

logger = logging.getLogger(__name__)

class MemorySummarizer:
    """
    Summarizes conversation history using the active LLM provider.
    """
    def __init__(self, llm: Any):
        self.llm = llm

    def generate_summary(self, existing_summary: str, turns: List[Turn]) -> str:
        """
        Merges the existing summary and new pruned turns into a concise summary.
        """
        if not turns:
            return existing_summary

        new_turns_text = ""
        for turn in turns:
            new_turns_text += f"User: {turn.user_message}\nAssistant: {turn.assistant_message}\n"

        prompt = (
            "You are a context compression assistant.\n"
            "Your task is to merge the existing conversation summary with a list of new conversation turns into a single, cohesive, and concise summary.\n"
            "Ensure all important user details, preferences, facts, and ongoing topics are preserved.\n"
            "Keep the summary under 3 sentences and write it in a conversational tone.\n\n"
            f"Existing summary:\n\"{existing_summary or 'No prior history.'}\"\n\n"
            f"New turns:\n{new_turns_text}\n"
            "Cohesive and concise summary:"
        )

        try:
            logger.info("Triggering LLM generation for rolling summary...")
            summary = self.llm.generate(prompt).strip()
            logger.info(f"New rolling summary generated successfully: '{summary}'")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback by appending a simple representation of the turns
            fallback_turns = "; ".join(f"U: {t.user_message} A: {t.assistant_message}" for t in turns)
            fallback = f"{existing_summary or ''} [History: {fallback_turns}]"
            return fallback[:400]
