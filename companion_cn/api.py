"""FastAPI router for the Chinese companion."""
import uuid
import os
import sqlite3
import json
import asyncio
import base64
import binascii
import re
import random
from contextlib import closing
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from .graph import graph
from .state import GraphState
from .config import QWEN_URL, QWEN_MODEL, QWEN_CHAT_TEMPLATE_KWARGS, DEEPSEEK_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL, USE_DEEPSEEK_GEN, DIALECT_STYLES, DEFAULT_CITY, DB_PATH
from .nodes import (
    input_guard, generate_safety_response, intent_detect, emotion_detect, memory_retrieve, tool_detect, context_assemble,
    generate_response, clean_and_remember,
    ground_visual_response, remove_virtual_actions as _remove_virtual_actions, EMOTION_PARAMS, FALLBACKS,
)
from .tools import format_current_official_answer
from .memory import list_facts, add_fact, delete_fact, clear_facts
from .reminders import (
    TZ, create_pending, activate_reminder, get_reminder, list_reminders, notified_reminders,
    complete_reminder as complete_reminder_by_id,
    snooze_reminder as snooze_reminder_by_id,
    cancel_reminder,
)
from .safety import scan
from .chat_record_reporter import report_session

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

router = APIRouter(prefix="/v1/langgraph", tags=["langgraph-cn"])

_model_client = OpenAI(base_url=QWEN_URL, api_key="x")
_ds_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL) if DEEPSEEK_KEY else None

_DB_PATH = DB_PATH
_CAMERA_DATA_URL_RE = re.compile(
    r"^data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)
_MAX_CAMERA_IMAGE_BYTES = 1_500_000
_AVATAR_ACTIONS = {
    "welcome", "encourage", "thanks", "comfort", "care", "stretch",
    "dance", "big_welcome", "celebrate", "big_comfort", "cheer",
    "full_stretch", "side_stretch", "march", "surprise", "goodnight",
}


def _chronic_heart_condition_reply(user_text: str) -> str | None:
    """Differentiate a stated heart condition from active emergency symptoms."""
    if "心脏病" not in user_text or any(
        phrase in user_text for phrase in ("没有心脏病", "没心脏病", "不是心脏病")
    ):
        return None
    return (
        "您刚才提到自己有心脏病。您现在有没有胸口疼或压迫感、喘不过气、冷汗、恶心、头晕，"
        "或者症状突然明显加重？如果有，请立即叫身边的人帮忙或拨打120；如果没有，我们慢慢说。"
    )


def _direct_empathy_reply(user_text: str) -> str | None:
    """Use stable, fact-grounded wording for clear emotional distress."""
    family_words = ("孙女", "孙子", "儿子", "女儿", "孩子", "家人")
    absence_words = ("没来看", "不来看", "没来", "不联系", "没联系", "没回", "不理我")
    family = next((word for word in family_words if word in user_text), "")
    if family and any(word in user_text for word in absence_words):
        return f"{family}一直没来看您，您一定很想念，心里也会空落落的。这样确实难受，我在这里听您说。"
    if any(word in user_text for word in ("不顺心", "委屈", "憋屈", "不是按照我的心意")):
        return "最近很多事都不顺您的心，您一定很憋屈，也很累。哪一件事最让您难受？"
    if any(word in user_text for word in ("伤心", "难过", "孤单", "孤独", "空落")):
        return "听到您说心里很难受，我很在意您的感受。您愿意说说，最让您难受的是什么吗？"
    return None


def _choose_avatar_action(
    reply: str,
    user_message: str,
    user_emotion: str,
    intensity: float,
) -> str:
    """Select a bundled avatar pose locally without adding another LLM call."""
    text = f"{user_message}\n{reply}".lower()

    # Emotional safety always wins over playful keyword matches.
    if user_emotion in {"sadness", "loneliness", "anxiety", "fear"}:
        return "big_comfort" if intensity >= 0.65 else "comfort"
    if user_emotion == "anger":
        return "care"

    keyword_actions = (
        ("goodnight", ("晚安", "睡觉", "睡了", "休息吧", "做个好梦")),
        ("march", ("原地踏步", "原地走", "踏步练习")),
        ("side_stretch", ("侧身伸展", "侧弯", "侧腰拉伸")),
        ("full_stretch", ("全身伸展", "全身拉伸", "伸展全身")),
        ("stretch", ("伸展", "拉伸", "活动一下", "做操", "散步", "走一走")),
        ("thanks", ("谢谢", "感谢", "多亏", "辛苦了")),
        ("celebrate", ("生日", "过节", "纪念日", "成功了", "得奖", "好消息")),
        ("dance", ("跳舞", "舞一曲", "跟着音乐", "唱歌跳舞")),
        ("cheer", ("加油", "打打气", "振作", "鼓鼓劲")),
        ("encourage", ("完成了", "做到了", "进步", "吃药了", "喝水了", "锻炼了")),
        ("big_welcome", ("好久不见", "终于见到", "又见面了")),
        ("welcome", ("挥手", "打招呼", "你好", "您好", "早上好", "下午好")),
        ("surprise", ("惊喜", "没想到", "真巧", "新发现")),
    )
    for action, keywords in keyword_actions:
        if any(keyword in text for keyword in keywords):
            return action

    if user_emotion == "surprise":
        return "surprise"
    if user_emotion in {"joy", "excited"}:
        return "celebrate" if intensity >= 0.75 else "encourage"
    if user_emotion == "nostalgia":
        return "thanks"
    return "care"


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


class CameraImage(BaseModel):
    data: str
    mime_type: str = "image/jpeg"
    captured_at_ms: int | None = None


class ChatRequest(BaseModel):
    user_id: str = "anonymous"
    session_id: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    message: str = ""
    dialect: str = "northern"
    images: list[CameraImage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str
    response_cleaned: str
    emotion: dict | None
    avatar_action: str
    risk_level: int
    facts_used: list[str]


class MemoryCreateRequest(BaseModel):
    user_id: str
    fact: str


class ReminderCreateRequest(BaseModel):
    user_id: str
    content: str
    due_at: datetime
    repeat_rule: str | None = ""


class ReminderActionRequest(BaseModel):
    user_id: str
    minutes: int = 10


def _camera_image_blocks(images: list[CameraImage]) -> list[dict]:
    """Validate one current camera frame and convert it to OpenAI content blocks."""
    if not images:
        return []
    if len(images) > 1:
        raise HTTPException(status_code=422, detail="每轮对话最多上传一张摄像头画面")

    data_url = images[0].data.strip()
    match = _CAMERA_DATA_URL_RE.fullmatch(data_url)
    if not match:
        raise HTTPException(status_code=422, detail="摄像头画面必须是 JPEG、PNG 或 WebP data URL")
    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail="摄像头画面的 Base64 数据无效") from error
    if not raw or len(raw) > _MAX_CAMERA_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="摄像头画面过大，请降低分辨率后重试")

    return [{
        "type": "image_url",
        "image_url": {"url": data_url},
    }]


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
        "visual_images": _camera_image_blocks(req.images),
        "risk_level": 0,
        "risk_label": None,
        "emotion": None,
        "intent": None,
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
    response = result.get("response") or ""
    cleaned = result.get("response_cleaned") or response
    tool_result = result.get("tool_result") or {}
    if tool_result.get("type") in ("reminder", "memory", "meal_history"):
        response = cleaned = tool_result.get("answer", "")
    elif result.get("risk_level", 0) < 3 and not result.get("visual_images"):
        chronic_reply = _chronic_heart_condition_reply(user_in)
        empathy_reply = _direct_empathy_reply(user_in)
        if chronic_reply:
            response = cleaned = chronic_reply
            em = {"primary": "neutral", "intensity": 0.4}
        elif empathy_reply:
            response = cleaned = empathy_reply
            em = {"primary": "sadness", "intensity": 0.7}
    response = _remove_virtual_actions(response)
    cleaned = _remove_virtual_actions(cleaned)
    return ChatResponse(
        user_id=req.user_id,
        session_id=sid,
        response=response,
        response_cleaned=cleaned,
        emotion={"primary": em["primary"], "intensity": em["intensity"]} if em and em.get("primary") else None,
        avatar_action=_choose_avatar_action(
            cleaned or response,
            user_in,
            em.get("primary", "neutral") if em else "neutral",
            em.get("intensity", 0) if em else 0,
        ),
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


@router.get("/memories/{user_id}")
async def memories(user_id: str):
    return {"user_id": user_id, "memories": list_facts(user_id)}


@router.post("/memories")
async def create_memory(req: MemoryCreateRequest):
    fact = req.fact.strip()
    if not fact:
        return {"ok": False, "error": "fact is required"}
    add_fact(req.user_id, fact, "user_confirmed")
    return {"ok": True, "fact": fact}


@router.delete("/memories/{user_id}/{fact_id}")
async def remove_memory(user_id: str, fact_id: int):
    return {"ok": delete_fact(user_id, fact_id)}


@router.delete("/memories/{user_id}")
async def remove_all_memories(user_id: str, confirm: bool = False):
    if not confirm:
        return {"ok": False, "error": "confirm=true is required"}
    return {"ok": True, "deleted": clear_facts(user_id)}


@router.get("/reminders/{user_id}")
async def reminders(user_id: str, include_finished: bool = False):
    return {"user_id": user_id, "reminders": list_reminders(user_id, include_finished)}


@router.get("/reminders/{user_id}/due")
async def due_reminders(user_id: str):
    """Frontend polling endpoint. Items remain until completed or snoozed."""
    return {"user_id": user_id, "reminders": notified_reminders(user_id)}


@router.post("/reminders")
async def create_reminder(req: ReminderCreateRequest):
    content = req.content.strip()
    _, risk_label, safe_reply = scan(content)
    if risk_label in ("self_harm", "violence"):
        # Reminders must never schedule or normalize physical harm.
        # Return the same supportive guidance used by the chat safety gate.
        return {
            "ok": False,
            "error": "无法创建涉及伤害自己或他人的提醒。",
            "response": (await asyncio.to_thread(generate_safety_response, {
                "user_input": content, "risk_label": risk_label, "response": safe_reply,
            })).get("response", safe_reply),
        }
    due_at = req.due_at.replace(tzinfo=TZ) if req.due_at.tzinfo is None else req.due_at.astimezone(TZ)
    reminder = create_pending(req.user_id, content, due_at, req.repeat_rule or "")
    reminder = activate_reminder(req.user_id, reminder["id"])
    return {"ok": True, "reminder": reminder}


@router.post("/reminders/{reminder_id}/confirm")
async def confirm_reminder(reminder_id: int, req: ReminderActionRequest):
    existing = get_reminder(req.user_id, reminder_id)
    if existing:
        _, risk_label, safe_reply = scan(existing["content"])
        if risk_label in ("self_harm", "violence"):
            cancel_reminder(req.user_id, reminder_id)
            return {
                "ok": False,
                "error": "无法确认涉及伤害自己或他人的提醒。",
                "response": (await asyncio.to_thread(generate_safety_response, {
                    "user_input": existing["content"], "risk_label": risk_label, "response": safe_reply,
                })).get("response", safe_reply),
            }
    reminder = activate_reminder(req.user_id, reminder_id)
    return {"ok": bool(reminder), "reminder": reminder}


@router.post("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: int, req: ReminderActionRequest):
    reminder = complete_reminder_by_id(req.user_id, reminder_id)
    return {"ok": bool(reminder), "reminder": reminder}


@router.post("/reminders/{reminder_id}/snooze")
async def snooze_reminder(reminder_id: int, req: ReminderActionRequest):
    updated = snooze_reminder_by_id(req.user_id, reminder_id, max(1, min(req.minutes, 180)))
    return {"ok": bool(updated), "reminder": updated}


@router.delete("/reminders/{reminder_id}")
async def remove_reminder(reminder_id: int, user_id: str):
    return {"ok": cancel_reminder(user_id, reminder_id)}


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
        "visual_images": _camera_image_blocks(req.images),
        "risk_level": 0, "risk_label": None, "emotion": None,
        "intent": None,
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
        safety = await asyncio.to_thread(generate_safety_response, state)
        async def safe_stream():
            safe = safety.get("response", state.get("response", random.choice(FALLBACKS)))
            # The web client renders tokens only before the terminal event.
            # Sending the safety text as `done: true` made it appear empty and
            # incorrectly fall back to the network-error message.
            yield f"data: {json.dumps({'token': safe, 'done': False, 'emotion': None}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': None, 'avatar_action': 'care', 'news': None, 'reminder': None}, ensure_ascii=False)}\n\n"
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    state.update(intent_detect(state))
    state.update(emotion_detect(state))
    state.update(memory_retrieve(state))
    state.update(tool_detect(state))
    state.update(context_assemble(state))

    tool_result = state.get("tool_result") or {}
    if tool_result.get("type") == "official":
        answer = format_current_official_answer(tool_result)

        async def official_stream():
            yield f"data: {json.dumps({'token': answer, 'done': False}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': None, 'avatar_action': _choose_avatar_action(answer, user_in, 'neutral', 0), 'news': None}, ensure_ascii=False)}\n\n"

        return StreamingResponse(official_stream(), media_type="text/event-stream")
    if tool_result.get("type") in ("reminder", "memory", "meal_history"):
        answer = tool_result.get("answer", "")
        reminder = None
        if tool_result.get("type") == "reminder" and tool_result.get("requires_confirmation"):
            reminder = {
                "id": tool_result.get("reminder_id"),
                "content": tool_result.get("content", ""),
                "due_at": tool_result.get("due_at", ""),
                "requires_confirmation": True,
            }

        async def managed_stream():
            yield f"data: {json.dumps({'token': answer, 'done': False}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': None, 'avatar_action': _choose_avatar_action(answer, user_in, 'neutral', 0), 'news': None, 'reminder': reminder}, ensure_ascii=False)}\n\n"

        return StreamingResponse(managed_stream(), media_type="text/event-stream")

    if not state.get("visual_images"):
        chronic_reply = _chronic_heart_condition_reply(user_in)
        empathy_reply = _direct_empathy_reply(user_in)
        if chronic_reply:
            async def chronic_stream():
                yield f"data: {json.dumps({'token': chronic_reply, 'done': False}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': {'primary': 'neutral', 'intensity': 0.4}, 'avatar_action': 'care', 'news': None, 'reminder': None}, ensure_ascii=False)}\n\n"
            return StreamingResponse(chronic_stream(), media_type="text/event-stream")
        if empathy_reply:
            async def empathy_stream():
                yield f"data: {json.dumps({'token': empathy_reply, 'done': False}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': {'primary': 'sadness', 'intensity': 0.7}, 'avatar_action': 'comfort', 'news': None, 'reminder': None}, ensure_ascii=False)}\n\n"
            return StreamingResponse(empathy_stream(), media_type="text/event-stream")

    msgs = state.get("messages", [])
    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    # Choose backend
    # DeepSeek Chat is text-only in this deployment. Camera turns must use the
    # configured multimodal Qwen endpoint so the image is not silently lost.
    if USE_DEEPSEEK_GEN and _ds_client and not state.get("visual_images"):
        stream_client = _ds_client
        stream_model = DEEPSEEK_MODEL
        stream_extra = {}
    else:
        stream_client = _model_client
        stream_model = QWEN_MODEL
        stream_extra = {
            "repetition_penalty": params["repetition_penalty"],
            "chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS,
        }

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
                max_tokens=220,
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
                        segment = _remove_virtual_actions(ground_visual_response(segment, state))
                        if not segment.strip():
                            continue
                        yield f"data: {json.dumps({'token': segment, 'done': False}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.12)

            if buffer.strip():
                final_segment = _remove_virtual_actions(ground_visual_response(buffer, state))
                if final_segment.strip():
                    yield f"data: {json.dumps({'token': final_segment, 'done': False}, ensure_ascii=False)}\n\n"

            full_response = _remove_virtual_actions(full_response)
            tool_result = state.get("tool_result") or {}
            news = None
            if tool_result.get("type") == "news" and tool_result.get("ok"):
                news = {
                    "query": user_in,
                    "article_ids": [article["id"] for article in tool_result.get("articles", [])],
                }
            intensity = em.get("intensity", 0) if em else 0
            avatar_action = _choose_avatar_action(
                full_response,
                user_in,
                primary,
                intensity,
            )
            yield f"data: {json.dumps({'token': '', 'done': True, 'emotion': {'primary': primary, 'intensity': intensity}, 'avatar_action': avatar_action, 'news': news}, ensure_ascii=False)}\n\n"

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
async def list_news(country: str = "", source: str = "", pinned_ids: str = "", limit: int = 50):
    """Browse scraped news from the knowledge base. Optional country/source filters."""
    import sqlite3
    _db = os.path.join(os.path.dirname(__file__), "data", "news.db")
    if not os.path.exists(_db):
        return {"ok": False, "error": "News database not found. Run news_scraper.py first."}
    with sqlite3.connect(_db) as conn:
        conn.row_factory = sqlite3.Row
        pin_ids = list(dict.fromkeys(
            int(value.strip()) for value in pinned_ids.split(",") if value.strip().isdigit()
        ))[:20]
        query = "SELECT id, title, summary, source, category, country, url, published_at, scraped_at FROM news_articles WHERE 1=1"
        params = []
        if country:
            query += " AND country LIKE ?"
            params.append(f"%{country}%")
        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")
        if pin_ids:
            placeholders = ",".join("?" for _ in pin_ids)
            query += f" ORDER BY CASE WHEN id IN ({placeholders}) THEN 0 ELSE 1 END, id DESC LIMIT ?"
            params.extend(pin_ids)
        else:
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


@router.post("/logs/{session_id}/complete", status_code=202)
async def complete_chat_session(
    session_id: str, background_tasks: BackgroundTasks
):
    """Accept the browser's end-of-chat signal and upload the saved session."""
    if not session_id or len(session_id) > 200:
        raise HTTPException(status_code=400, detail="invalid session_id")

    with closing(sqlite3.connect(_DB_PATH)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM conversations_cn WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="chat session not found")

    background_tasks.add_task(report_session, session_id)
    return {"accepted": True, "session_id": session_id}


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
