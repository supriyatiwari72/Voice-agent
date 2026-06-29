from memory.models import Turn
from memory.context_builder import ContextBuilder

def test_context_builder_full():
    """
    Verify prompt contains system prompt, summary, facts, recent turns, and current query.
    """
    system_prompt = "Keep responses short."
    summary = "User discussed AI agent architectures."
    facts = {"name": "Supriya", "project": "Voice Agent"}
    recent_turns = [
        Turn("hello", "hi"),
        Turn("how are you?", "great")
    ]
    user_query = "What is my project name?"
    
    context = ContextBuilder.build_context(
        system_prompt=system_prompt,
        summary=summary,
        facts=facts,
        recent_turns=recent_turns,
        user_query=user_query
    )
    
    assert "[System Prompt]\nKeep responses short." in context
    assert "[Conversation Summary]\nUser discussed AI agent architectures." in context
    assert "[User Profile Facts]\n- name: Supriya\n- project: Voice Agent" in context
    assert "[Recent Turns]\nUser: hello\nAssistant: hi\nUser: how are you?\nAssistant: great" in context
    assert "User: What is my project name?" in context

def test_context_builder_empty_states():
    """
    Verify builder handles empty fields gracefully without throwing errors.
    """
    context = ContextBuilder.build_context(
        system_prompt="",
        summary="",
        facts={},
        recent_turns=[],
        user_query="hello"
    )
    
    assert "[System Prompt]" not in context
    assert "[Conversation Summary]" not in context
    assert "[User Profile Facts]" not in context
    assert "[Recent Turns]" not in context
    assert context.strip() == "User: hello"
