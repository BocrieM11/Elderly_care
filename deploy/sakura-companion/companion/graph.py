"""LangGraph StateGraph — parallel fan-out + quality loop + checkpoint persistence."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from .state import GraphState
from .nodes import (
    input_guard, emotion_detect, memory_retrieve, context_assemble,
    generate_response, clean_and_translate_and_remember,
)

def _after_guard(state: GraphState) -> list[str]:
    """Fan-out: run emotion_detect and memory_retrieve in parallel, or short-circuit."""
    if state.get("risk_level", 0) >= 3:
        return [END]
    return ["emotion_detect", "memory_retrieve"]


def build() -> StateGraph:
    g = StateGraph(GraphState)
    g.add_node("input_guard", input_guard)
    g.add_node("emotion_detect", emotion_detect)
    g.add_node("memory_retrieve", memory_retrieve)
    g.add_node("context_assemble", context_assemble)
    g.add_node("generate_response", generate_response)
    g.add_node("clean_translate_remember", clean_and_translate_and_remember)

    g.set_entry_point("input_guard")

    # input_guard → fan-out to [emotion_detect ‖ memory_retrieve] in parallel
    g.add_conditional_edges("input_guard", _after_guard, {
        "emotion_detect": "emotion_detect",
        "memory_retrieve": "memory_retrieve",
        END: END,
    })

    # Both converge → context_assemble (waits for both to finish)
    g.add_edge("emotion_detect", "context_assemble")
    g.add_edge("memory_retrieve", "context_assemble")

    g.add_edge("context_assemble", "generate_response")
    g.add_edge("generate_response", "clean_translate_remember")
    g.add_edge("clean_translate_remember", END)

    return g.compile(checkpointer=InMemorySaver())


graph = build()
