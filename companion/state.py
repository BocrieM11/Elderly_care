"""Graph state — single TypedDict used by every node."""
from typing import TypedDict, Optional


class GraphState(TypedDict):
    user_id: str
    session_id: str
    user_input: str
    messages: list  # OpenAI-format [{"role":"...","content":"..."}]

    # Safety
    risk_level: int                 # 0-4
    risk_label: Optional[str]

    # Emotion
    emotion: Optional[dict]

    # Memory
    memory_facts: list[str]         # facts about this user from SQLite

    # Generation
    system_prompt: str
    response: Optional[str]
    response_cleaned: Optional[str]
    translation: Optional[str]

    # Flow
    error: Optional[str]
    temperature: float              # dynamic temperature from emotion
