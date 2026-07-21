"""Graph state — single TypedDict used by every node."""
from typing import TypedDict, Optional


class GraphState(TypedDict):
    user_id: str
    session_id: str
    user_input: str
    messages: list  # OpenAI-format [{"role":"...","content":"..."}]
    visual_images: list[dict]  # Current-turn OpenAI image_url blocks from the camera

    # Safety
    risk_level: int                 # 0-4
    risk_label: Optional[str]

    # Emotion
    emotion: Optional[dict]
    intent: Optional[dict]

    # Memory
    memory_facts: list[str]         # facts about this user from SQLite

    # Generation
    system_prompt: str
    response: Optional[str]
    response_cleaned: Optional[str]

    # Flow
    error: Optional[str]
    temperature: float              # dynamic temperature from emotion
    topic_summary: Optional[str]    # one-line topic summary, refreshed every ~3 turns
    asked_questions: list[str]      # questions already asked (dedup)
    dialect: Optional[str]          # colloquial style: northern | dongbei | sichuan | jiangzhe | standard
    tool_result: Optional[dict]     # {type, text, ok} from tools.py
    weather_city: Optional[str]     # city for weather queries, default "北京"
