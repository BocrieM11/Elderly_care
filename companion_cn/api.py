"""FastAPI router for the Chinese companion."""
import uuid
import os
import sqlite3
import json
import asyncio
import re
import random
from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from .graph import graph
from .state import GraphState
from .config import QWEN_URL, QWEN_MODEL, DEEPSEEK_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL, USE_DEEPSEEK_GEN, DIALECT_STYLES, DEFAULT_CITY
from .nodes import (
    input_guard, emotion_detect, memory_retrieve, tool_detect, context_assemble,
    generate_response, clean_and_remember,
    EMOTION_PARAMS, FALLBACKS,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

router = APIRouter(prefix="/v1/langgraph", tags=["langgraph-cn"])

_model_client = OpenAI(base_url=QWEN_URL, api_key="x")
_ds_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL) if DEEPSEEK_KEY else None

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "arrowcanaria_server", "chat_logs.db")


def _init_db():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations_cn (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt_type TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                reply TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_cn ON conversations_cn(session_id, round_number)")
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
    dialect: str = "northern"


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
        "error": None,
        "temperature": 0.75,
        "dialect": req.dialect,
        "tool_result": None,
        "weather_city": DEFAULT_CITY,
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
    """Streaming chat for Chinese companion."""
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
        "error": None,
        "temperature": 0.75,
        "dialect": req.dialect,
        "tool_result": None,
        "weather_city": DEFAULT_CITY,
    }

    # Pre-processing
    guard = input_guard(state)
    state.update(guard)
    if state.get("risk_level", 0) >= 3:
        async def safe_stream():
            safe = state.get("response", random.choice(FALLBACKS))
            yield f"data: {json.dumps({'token': safe, 'done': True, 'emotion': None}, ensure_ascii=False)}\n\n"
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    state.update(emotion_detect(state))
    state.update(memory_retrieve(state))
    state.update(tool_detect(state))
    state.update(context_assemble(state))

    msgs = state.get("messages", [])
    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    # Choose backend
    if USE_DEEPSEEK_GEN and _ds_client:
        stream_client = _ds_client
        stream_model = DEEPSEEK_MODEL
        stream_extra = {}
    else:
        stream_client = _model_client
        stream_model = QWEN_MODEL
        stream_extra = {"repetition_penalty": params["repetition_penalty"]}

    async def event_stream():
        full_response = ""
        buffer = ""
        SENTENCE_END = re.compile(r"[。？！…\n]")
        try:
            stream = stream_client.chat.completions.create(
                model=stream_model,
                messages=msgs,
                temperature=params["temperature"],
                top_p=params.get("top_p", 0.88),
                frequency_penalty=params.get("frequency_penalty", 0.1),
                presence_penalty=params.get("presence_penalty", 0.0),
                max_tokens=350,
                stream=True,
                extra_body=stream_extra,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
                if delta:
                    full_response += delta
                    buffer += delta
                    m = SENTENCE_END.search(buffer)
                    if m:
                        end = m.end()
                        after = buffer[end:]
                        if after and re.match(r'^[。、！？…\s]{1,3}$', after):
                            end += len(after)
                        segment = buffer[:end]
                        if len(segment.strip()) < 4:
                            continue
                        buffer = buffer[end:]
                        yield f"data: {json.dumps({'token': segment, 'done': False}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.12)

            if buffer.strip():
                yield f"data: {json.dumps({'token': buffer, 'done': False}, ensure_ascii=False)}\n\n"

            # Done event with emotion
            yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': {'primary': primary, 'intensity': em.get('intensity', 0) if em else 0}}, ensure_ascii=False)}\n\n"

            # Post-processing: memory extraction (async after done)
            state["response"] = full_response
            clean_and_remember(state)

        except Exception as e:
            yield f"data: {json.dumps({'token': '', 'done': True, 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/dialects")
async def list_dialects():
    """Return available colloquial styles for the frontend selector."""
    return {"dialects": {k: {"label": v["label"]} for k, v in DIALECT_STYLES.items()}}

@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/news/check")
async def check_news(q: str = ""):
    """Quick search: does a keyword/claim exist in the news DB?
    Use this to verify if the model's reply is based on real news.
    Example: /v1/langgraph/news/check?q=潜射导弹"""
    import sqlite3
    _db = os.path.join(os.path.dirname(__file__), "data", "news.db")
    if not q or not os.path.exists(_db):
        return {"ok": False, "found": False, "message": "数据库为空或未提供查询词"}
    with sqlite3.connect(_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT title, source, url FROM news_articles
               WHERE title LIKE ? OR summary LIKE ?
               ORDER BY id DESC LIMIT 5""",
            (f"%{q}%", f"%{q}%"),
        ).fetchall()
        if rows:
            return {
                "ok": True, "found": True, "query": q,
                "results": [{"title": r["title"], "source": r["source"], "url": r["url"]} for r in rows],
            }
        return {"ok": True, "found": False, "query": q, "message": f"数据库中未找到与「{q}」相关的新闻"}


@router.get("/news")
async def list_news(country: str = "", source: str = "", limit: int = 50):
    """Browse scraped news from the knowledge base. Optional country/source filters."""
    import sqlite3
    _db = os.path.join(os.path.dirname(__file__), "data", "news.db")
    if not os.path.exists(_db):
        return {"ok": False, "error": "News database not found. Run news_scraper.py first."}
    with sqlite3.connect(_db) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT id, title, summary, source, category, country, url, published_at, scraped_at FROM news_articles WHERE 1=1"
        params = []
        if country:
            query += " AND country LIKE ?"
            params.append(f"%{country}%")
        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return {
            "ok": True,
            "count": len(rows),
            "articles": [dict(r) for r in rows],
        }


# ---------------------------------------------------------------------------
# Conversation logging
# ---------------------------------------------------------------------------
class LogRequest(BaseModel):
    session_id: str
    prompt_type: str
    round_number: int
    user_message: str
    sakura_reply: str


@router.post("/log")
async def save_log(req: LogRequest):
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO conversations_cn (session_id, prompt_type, round_number, user_message, reply, created_at) VALUES (?,?,?,?,?,?)",
                (req.session_id, req.prompt_type, req.round_number, req.user_message, req.sakura_reply, datetime.now(timezone.utc).isoformat())
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
            FROM conversations_cn
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
            "SELECT * FROM conversations_cn WHERE session_id=? ORDER BY round_number",
            (session_id,)
        ).fetchall()
        return {"session_id": session_id, "rounds": [dict(r) for r in rows]}


@router.delete("/logs/{session_id}")
async def delete_session(session_id: str):
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DELETE FROM conversations_cn WHERE session_id=?", (session_id,))
        conn.commit()
    return {"ok": True}
