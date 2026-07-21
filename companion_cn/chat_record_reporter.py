"""Upload completed Companion CN sessions to the resident analysis service."""

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone

import requests

from .config import (
    CHAT_RECORD_API_TOKEN,
    CHAT_RECORD_ENABLED,
    CHAT_RECORD_ENDPOINT,
    CHAT_RECORD_MAX_RETRIES,
    CHAT_RECORD_RESIDENT_NAME,
    CHAT_RECORD_ROOM_NO,
    CHAT_RECORD_TIMEOUT_SECONDS,
    DB_PATH,
)


logger = logging.getLogger(__name__)
REPORT_TABLE = "chat_record_reports"
LOCAL_TIMEZONE = timezone(timedelta(hours=8))
_report_lock = threading.Lock()


def init_report_table(db_path: str = DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {REPORT_TABLE} (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                sent_at TEXT
            )
            """
        )
        conn.commit()


def _now_iso() -> str:
    return datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds")


def _to_local_iso(value: str | None) -> str:
    if value:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
            return parsed.astimezone(LOCAL_TIMEZONE).isoformat(timespec="seconds")
        except ValueError:
            logger.warning("Invalid conversation timestamp: %s", value)
    return _now_iso()


def build_session_payload(
    session_id: str,
    db_path: str = DB_PATH,
    resident_name: str = CHAT_RECORD_RESIDENT_NAME,
    room_no: str = CHAT_RECORD_ROOM_NO,
) -> dict | None:
    """Build the receiver payload from authoritative saved conversation rounds."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT user_message, reply, created_at
            FROM conversations_cn
            WHERE session_id = ?
            ORDER BY round_number, id
            """,
            (session_id,),
        ).fetchall()

    if not rows:
        return None

    messages = []
    for row in rows:
        user_message = (row["user_message"] or "").strip()
        reply = (row["reply"] or "").strip()
        if user_message:
            messages.append({"role": "老人", "content": user_message})
        if reply:
            messages.append({"role": "AI", "content": reply})

    if not messages:
        return None

    payload = {
        "session_id": session_id,
        "timestamp": _to_local_iso(rows[0]["created_at"]),
        "session_ended_at": _now_iso(),
        "source": "Companion-CN",
        "messages": messages,
    }
    if resident_name.strip():
        payload["resident_name"] = resident_name.strip()
    if room_no.strip():
        payload["room_no"] = room_no.strip()
    return payload


def _get_report_status(session_id: str, db_path: str) -> str | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            f"SELECT status FROM {REPORT_TABLE} WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row[0] if row else None


def _save_report_status(
    session_id: str,
    status: str,
    message_count: int,
    attempts: int,
    last_error: str,
    db_path: str,
) -> None:
    now = _now_iso()
    sent_at = now if status == "sent" else None
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f"""
            INSERT INTO {REPORT_TABLE} (
                session_id, status, message_count, attempts, last_error,
                updated_at, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status = excluded.status,
                message_count = excluded.message_count,
                attempts = excluded.attempts,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at,
                sent_at = excluded.sent_at
            """,
            (
                session_id,
                status,
                message_count,
                attempts,
                last_error[:500],
                now,
                sent_at,
            ),
        )
        conn.commit()


def report_session_sync(
    session_id: str,
    db_path: str = DB_PATH,
    endpoint: str = CHAT_RECORD_ENDPOINT,
    enabled: bool = CHAT_RECORD_ENABLED,
    timeout_seconds: float = CHAT_RECORD_TIMEOUT_SECONDS,
    max_retries: int = CHAT_RECORD_MAX_RETRIES,
    api_token: str = CHAT_RECORD_API_TOKEN,
) -> bool:
    """Upload a completed session once and persist its delivery status."""
    if not enabled or not session_id:
        return False

    init_report_table(db_path)
    with _report_lock:
        if _get_report_status(session_id, db_path) == "sent":
            logger.info("Chat session %s was already reported", session_id)
            return True

        payload = build_session_payload(session_id, db_path=db_path)
        if payload is None:
            logger.info("Skipping empty chat session %s", session_id)
            return False

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["X-Service-Token"] = api_token

        attempts = max_retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                _save_report_status(
                    session_id,
                    "sent",
                    len(payload["messages"]),
                    attempt,
                    "",
                    db_path,
                )
                logger.info(
                    "Reported Companion chat session %s (%s messages)",
                    session_id,
                    len(payload["messages"]),
                )
                return True
            except requests.RequestException as error:
                last_error = str(error)
                logger.warning(
                    "Failed to report Companion session %s (attempt %s/%s): %s",
                    session_id,
                    attempt,
                    attempts,
                    error,
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))

        _save_report_status(
            session_id,
            "pending",
            len(payload["messages"]),
            attempts,
            last_error,
            db_path,
        )
        return False


async def report_session(session_id: str) -> bool:
    """Run blocking SQLite and HTTP work outside the FastAPI event loop."""
    return await asyncio.to_thread(report_session_sync, session_id)


def pending_session_ids(db_path: str = DB_PATH) -> list[str]:
    init_report_table(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT session_id FROM {REPORT_TABLE} WHERE status = 'pending'"
        ).fetchall()
    return [row[0] for row in rows]


async def retry_pending_sessions() -> None:
    """Retry durable pending reports once when the Companion service starts."""
    for session_id in await asyncio.to_thread(pending_session_ids):
        await report_session(session_id)


init_report_table()
