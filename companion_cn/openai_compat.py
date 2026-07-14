"""OpenAI-compatible API wrapper for the companion service.

Any OpenAI SDK client can point to http://<host>:8016/v1 and get
companion-enhanced responses (emotion detection, safety, memory, dialect).

Usage:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8016/v1", api_key="x")
    client.chat.completions.create(
        model="companion-cn",
        messages=[{"role": "user", "content": "最近腰老是疼"}],
        extra_body={"dialect": "dongbei"},
    )
"""
import uuid
import json
import time
import re
import random
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal
from openai import OpenAI

from .config import QWEN_URL, QWEN_MODEL, QWEN_CHAT_TEMPLATE_KWARGS, DEEPSEEK_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL, USE_DEEPSEEK_GEN, DIALECT_STYLES
from .nodes import (
    input_guard, emotion_detect, memory_retrieve, context_assemble,
    generate_response, clean_and_remember,
    EMOTION_PARAMS, FALLBACKS,
)
from .state import GraphState

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_model_client = OpenAI(base_url=QWEN_URL, api_key="x")
_ds_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL) if DEEPSEEK_KEY else None


# ── Pydantic models (OpenAI format) ──────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "companion-cn"
    messages: list[ChatMessage]
    temperature: Optional[float] = None       # overridden by emotion detection
    max_tokens: Optional[int] = 220
    stream: Optional[bool] = False
    user: Optional[str] = None                # maps to user_id
    # ── Companion-specific extensions ──
    session_id: Optional[str] = None
    dialect: Optional[str] = "northern"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_chat_completion(
    content: str,
    model: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens or len(content),
            "total_tokens": (prompt_tokens + completion_tokens or len(content)),
        },
    }


def _build_chunk(content: str, model: str, finish_reason: Optional[str] = None) -> dict:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    if finish_reason is None:
        chunk["choices"][0]["delta"]["content"] = content
    return chunk


def _build_state(req: ChatCompletionRequest) -> tuple[GraphState, str, str]:
    """Convert OpenAI request to internal state. Returns (state, user_input, sid)."""
    user_in = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_in = m.content
            break

    sid = req.session_id or uuid.uuid4().hex[:12]
    uid = req.user or "anonymous"

    state: GraphState = {
        "user_id": uid,
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
        "temperature": req.temperature or 0.75,
        "topic_summary": "",
        "asked_questions": [],
        "dialect": req.dialect or "northern",
    }
    return state, user_in, sid


# ── Non-streaming endpoint ───────────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions. Wraps the full companion pipeline."""
    state, user_in, sid = _build_state(req)

    # --- Safety gate ---
    guard = input_guard(state)
    state.update(guard)
    if state.get("risk_level", 0) >= 3:
        safe = state.get("response", random.choice(FALLBACKS))
        return _build_chat_completion(safe, req.model, "stop", 0, len(safe))

    # --- Full pipeline ---
    state.update(emotion_detect(state))
    state.update(memory_retrieve(state))
    state.update(context_assemble(state))

    msgs = state.get("messages", [])
    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    stream_client = _ds_client if (USE_DEEPSEEK_GEN and _ds_client) else _model_client
    stream_model = DEEPSEEK_MODEL if (USE_DEEPSEEK_GEN and _ds_client) else QWEN_MODEL
    extra = {} if (USE_DEEPSEEK_GEN and _ds_client) else {
        "repetition_penalty": params["repetition_penalty"],
        "chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS,
    }

    try:
        r = stream_client.chat.completions.create(
            model=stream_model,
            messages=msgs,
            temperature=params["temperature"],
            top_p=params.get("top_p", 0.88),
            frequency_penalty=params.get("frequency_penalty", 0.1),
            presence_penalty=params.get("presence_penalty", 0.0),
            max_tokens=min(req.max_tokens or 220, 220),
            extra_body=extra,
        )
        raw = r.choices[0].message.content or ""
    except Exception as e:
        return _build_chat_completion(
            random.choice(FALLBACKS), req.model, "error", 0, 0
        )

    # Post-processing
    state["response"] = raw
    clean_state = clean_and_remember(state)
    cleaned = clean_state.get("response_cleaned", raw)

    return _build_chat_completion(cleaned, req.model, "stop", 0, len(cleaned))


# ── Streaming endpoint ───────────────────────────────────────────────────────

@router.post("/chat/completions/stream")
async def chat_completions_stream(req: ChatCompletionRequest):
    """OpenAI-compatible streaming chat completions (SSE)."""
    state, user_in, sid = _build_state(req)

    # --- Safety gate ---
    guard = input_guard(state)
    state.update(guard)
    if state.get("risk_level", 0) >= 3:
        async def safe_stream():
            safe = state.get("response", random.choice(FALLBACKS))
            chunk = json.dumps(_build_chunk(safe, req.model, "stop"), ensure_ascii=False)
            yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    # --- Full pipeline ---
    state.update(emotion_detect(state))
    state.update(memory_retrieve(state))
    state.update(context_assemble(state))

    msgs = state.get("messages", [])
    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    stream_client = _ds_client if (USE_DEEPSEEK_GEN and _ds_client) else _model_client
    stream_model = DEEPSEEK_MODEL if (USE_DEEPSEEK_GEN and _ds_client) else QWEN_MODEL
    extra = {} if (USE_DEEPSEEK_GEN and _ds_client) else {
        "repetition_penalty": params["repetition_penalty"],
        "chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS,
    }

    async def event_stream():
        full_response = ""
        try:
            # Send initial chunk with role
            init = json.dumps({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }, ensure_ascii=False)
            yield f"data: {init}\n\n"

            stream = stream_client.chat.completions.create(
                model=stream_model,
                messages=msgs,
                temperature=params["temperature"],
                top_p=params.get("top_p", 0.88),
                frequency_penalty=params.get("frequency_penalty", 0.1),
                presence_penalty=params.get("presence_penalty", 0.0),
                max_tokens=min(req.max_tokens or 220, 220),
                stream=True,
                extra_body=extra,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
                if delta:
                    full_response += delta
                    chunk_json = json.dumps(_build_chunk(delta, req.model), ensure_ascii=False)
                    yield f"data: {chunk_json}\n\n"

            # Done
            done_chunk = json.dumps(_build_chunk("", req.model, "stop"), ensure_ascii=False)
            yield f"data: {done_chunk}\n\n"
            yield "data: [DONE]\n\n"

            # Post-processing
            state["response"] = full_response
            clean_and_remember(state)

        except Exception as e:
            error_chunk = json.dumps({
                **{k: v for k, v in _build_chunk("", req.model, "error").items() if k != "choices"},
                "error": str(e),
            }, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Models list (OpenAI compatibility) ───────────────────────────────────────

@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "companion-cn",
                "object": "model",
                "created": 1750000000,
                "owned_by": "companion",
            }
        ],
    }
