"""FastAPI router for the LangGraph companion."""
import uuid, os, sqlite3, json, asyncio, time, re
from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from .graph import graph
from .state import GraphState
from .config import ARROWCANARIA_URL
from .nodes import (
    input_guard, emotion_detect, memory_retrieve, context_assemble,
    generate_response, clean_and_translate_and_remember,
    EMOTION_PARAMS, EMOTION_TO_TAG, FALLBACK,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

router = APIRouter(prefix="/v1/langgraph", tags=["langgraph"])

# ArrowCanaria client for streaming
_model_client = OpenAI(base_url=ARROWCANARIA_URL, api_key="x")


# SQLite (shared DB with ArrowCanaria server)
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "arrowcanaria_server", "chat_logs.db")

def _init_db():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt_type TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                sakura_reply TEXT NOT NULL,
                translation TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id, round_number)")
        conn.commit()

_init_db()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str = "anonymous"
    session_id: str = ""
    messages: list[ChatMessage] = []
    message: str = ""


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str
    response_cleaned: str
    emotion: dict | None
    risk_level: int
    facts_used: list[str]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sid = req.session_id or uuid.uuid4().hex[:12]
    user_in = req.message
    if not user_in and req.messages:
        for m in reversed(req.messages):
            if m.role == "user":
                user_in = m.content
                break

    state: GraphState = {
        "user_id": req.user_id,
        "session_id": sid,
        "user_input": user_in,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "risk_level": 0,
        "risk_label": None,
        "emotion": None,
        "memory_facts": [],
        "system_prompt": "",
        "response": None,
        "response_cleaned": None,
        "translation": None,
        "error": None,
        "temperature": 0.75,
    }

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": sid}})

    em = result.get("emotion")
    return ChatResponse(
        user_id=req.user_id,
        session_id=sid,
        response=result.get("response") or "",
        response_cleaned=result.get("response_cleaned") or "",
        emotion={"primary": em["primary"], "intensity": em["intensity"]} if em and em.get("primary") else None,
        risk_level=result.get("risk_level", 0),
        facts_used=result.get("memory_facts", []),
    )


@router.get("/profile/{user_id}")
async def profile(user_id: str):
    from .memory import get_profile, get_facts
    return {
        "user_id": user_id,
        "profile": get_profile(user_id),
        "facts": get_facts(user_id, limit=30),
    }


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat: LangGraph pre-processing → ArrowCanaria SSE stream → post-processing."""
    sid = req.session_id or uuid.uuid4().hex[:12]
    user_in = req.message
    if not user_in and req.messages:
        for m in reversed(req.messages):
            if m.role == "user":
                user_in = m.content
                break

    state: GraphState = {
        "user_id": req.user_id,
        "session_id": sid,
        "user_input": user_in,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "risk_level": 0, "risk_label": None, "emotion": None,
        "memory_facts": [], "system_prompt": "",
        "response": None, "response_cleaned": None,
        "translation": None, "error": None,
        "temperature": 0.75,
    }

    # --- Pre-processing (LangGraph nodes, no LLM calls) ---
    # input_guard
    guard = input_guard(state)
    state.update(guard)
    if state.get("risk_level", 0) >= 3:
        async def safe_stream():
            safe = state.get("response", FALLBACK)
            yield f"data: {json.dumps({'token': safe, 'done': True, 'translation': '', 'emotion': None}, ensure_ascii=False)}\n\n"
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    # emotion_detect ‖ memory_retrieve (run sequentially — fast enough)
    state.update(emotion_detect(state))
    state.update(memory_retrieve(state))

    # context_assemble — builds enhanced prompt with PERSONA + memory
    state.update(context_assemble(state))

    msgs = state.get("messages", [])
    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    recommended_tag = EMOTION_TO_TAG.get(primary, "calm")
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    async def event_stream():
        full_response = ""
        buffer = ""
        SENTENCE_END = re.compile(r"[。？！…\n]")
        try:
            stream = _model_client.chat.completions.create(
                model="arrowcanaria-8b",
                messages=msgs,
                temperature=params["temperature"],
                max_tokens=120,
                stream=True,
                extra_body={"repetition_penalty": params["repetition_penalty"]},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
                if delta:
                    full_response += delta
                    buffer += delta
                    # 遇到句子边界就发送整段（段太短则继续累积）
                    m = SENTENCE_END.search(buffer)
                    if m:
                        end = m.end()
                        # 如果边界之后只剩短标点，扩展边界吃掉它
                        after = buffer[end:]
                        if after and re.match(r'^[。、！？…\s]{1,3}$', after):
                            end += len(after)
                        segment = buffer[:end]
                        # 段本身太短则继续累积
                        if len(segment.strip()) < 4:
                            continue
                        buffer = buffer[end:]
                        yield f"data: {json.dumps({'token': segment, 'done': False}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.15)

            # 发送剩余缓冲区
            if buffer.strip():
                yield f"data: {json.dumps({'token': buffer, 'done': False}, ensure_ascii=False)}\n\n"

            # --- Done event: send immediately with tag (fast) ---
            yield f"data: {json.dumps({'token': '', 'done': True, 'tag': recommended_tag, 'emotion': {'primary': primary, 'intensity': em.get('intensity', 0) if em else 0}}, ensure_ascii=False)}\n\n"

            # --- Post-processing: translation + memory (slow, async after done) ---
            state["response"] = full_response
            cleaned = clean_and_translate_and_remember(state)
            translation = cleaned.get("translation", "")
            if translation:
                yield f"data: {json.dumps({'translation': translation}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': '', 'done': True, 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Translation (DeepSeek)
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str


@router.post("/translate")
async def translate(req: TranslateRequest):
    try:
        if not _ds:
            return {"translation": "", "error": "DeepSeek key not configured"}
        r = _ds.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"将以下日语直接翻译成中文：\n{req.text}"}],
            temperature=0.3, max_tokens=200, timeout=10,
        )
        return {"translation": r.choices[0].message.content.strip()}
    except Exception as e:
        return {"translation": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Conversation logging (SQLite)
# ---------------------------------------------------------------------------
class LogRequest(BaseModel):
    session_id: str
    prompt_type: str
    round_number: int
    user_message: str
    sakura_reply: str
    translation: str = ""


@router.post("/log")
async def save_log(req: LogRequest):
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, prompt_type, round_number, user_message, sakura_reply, translation, created_at) VALUES (?,?,?,?,?,?,?)",
                (req.session_id, req.prompt_type, req.round_number, req.user_message, req.sakura_reply, req.translation, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/logs")
async def list_sessions():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT session_id, prompt_type, COUNT(*) as rounds,
                   MIN(created_at) as started, MAX(created_at) as last_active,
                   MIN(user_message) as first_msg
            FROM conversations
            GROUP BY session_id
            ORDER BY last_active DESC
            LIMIT 50
        """).fetchall()
        return {"sessions": [dict(r) for r in rows]}


@router.get("/logs/{session_id}")
async def get_session(session_id: str):
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM conversations WHERE session_id=? ORDER BY round_number",
            (session_id,)
        ).fetchall()
        return {"session_id": session_id, "rounds": [dict(r) for r in rows]}


@router.delete("/logs/{session_id}")
async def delete_session(session_id: str):
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        conn.commit()
    return {"ok": True}
