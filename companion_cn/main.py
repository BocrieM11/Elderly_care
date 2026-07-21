"""Chinese companion — FastAPI entry point (:8016)."""
import sys
import io
import os
import asyncio

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from .api import router
from .openai_compat import router as openai_router
from .media_api import router as media_router
from .config import PORT
from .reminders import init as init_reminders, mark_due_reminders
from .chat_record_reporter import retry_pending_sessions

app = FastAPI(title="养老陪伴", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_reminder_task = None
_chat_report_retry_task = None


async def _reminder_loop():
    """Mark due reminders in the database; delivery is handled by the web client."""
    while True:
        mark_due_reminders()
        await asyncio.sleep(15)


async def _chat_report_retry_loop():
    """Keep retrying durable pending uploads without blocking chat requests."""
    while True:
        await retry_pending_sessions()
        await asyncio.sleep(60)


@app.on_event("startup")
async def start_reminder_service():
    global _reminder_task, _chat_report_retry_task
    init_reminders()
    _reminder_task = asyncio.create_task(_reminder_loop())
    _chat_report_retry_task = asyncio.create_task(_chat_report_retry_loop())


@app.on_event("shutdown")
async def stop_reminder_service():
    if _reminder_task:
        _reminder_task.cancel()
    if _chat_report_retry_task:
        _chat_report_retry_task.cancel()

# Internal API (legacy)
app.include_router(router)
# OpenAI-compatible API (for external clients)
app.include_router(openai_router)
# Browser-facing ASR/TTS bridge for the Companion interface
app.include_router(media_router)

# Serve frontend (must be last to avoid route conflicts)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    print(f"养老陪伴服务启动 → http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
