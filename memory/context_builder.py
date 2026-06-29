from typing import List, Dict, Any
from memory.models import Turn

class ContextBuilder:
    """
    Constructs memory-enriched prompts for LLM execution.
    """
    @staticmethod
    def build_context(
        system_prompt: str,
        summary: str,
        facts: Dict[str, Any],
        recent_turns: List[Turn],
        user_query: str
    ) -> str:
        """
        Builds a comprehensive prompt containing system prompt, summary, facts, recent turns, and current query.
        """
        parts = []
        
        # System Prompt section
        if system_prompt:
            parts.append(f"[System Prompt]\n{system_prompt.strip()}")
            
        # Summary section
        if summary:
            parts.append(f"[Conversation Summary]\n{summary.strip()}")
            
        # Facts section
        if facts:
            facts_list = [f"- {k}: {v}" for k, v in facts.items()]
            if facts_list:
                parts.append("[User Profile Facts]\n" + "\n".join(facts_list))
                
        # Recent turns section
        if recent_turns:
            turns_text = []
            for turn in recent_turns:
                turns_text.append(f"User: {turn.user_message}")
                turns_text.append(f"Assistant: {turn.assistant_message}")
            parts.append("[Recent Turns]\n" + "\n".join(turns_text))
            
        # Current user query
        parts.append(f"User: {user_query.strip()}")
        
        return "\n\n".join(parts)
