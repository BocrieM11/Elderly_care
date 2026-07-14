"""Persistent, confirmation-first reminders for the companion."""
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from .config import STATE_DB_PATH

# The bundled Windows Python does not include IANA tzdata. China Standard Time
# has no daylight-saving transition, so an explicit UTC+8 zone is reliable.
TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _conn():
    import os
    os.makedirs(os.path.dirname(STATE_DB_PATH), exist_ok=True)
    db = sqlite3.connect(STATE_DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init():
    with _conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                due_at TEXT NOT NULL,
                repeat_rule TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending_confirmation',
                notified_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_id, status)")


def _now() -> datetime:
    return datetime.now(TZ)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _as_dict(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["due_at"] = datetime.fromisoformat(item["due_at"]).astimezone(TZ).isoformat()
    return item


def create_pending(user_id: str, content: str, due_at: datetime, repeat_rule: str = "") -> dict:
    now = _iso(_now())
    with _conn() as db:
        cur = db.execute(
            """INSERT INTO reminders(user_id, content, due_at, repeat_rule, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending_confirmation', ?, ?)""",
            (user_id, content, _iso(due_at), repeat_rule, now, now),
        )
        row = db.execute("SELECT * FROM reminders WHERE id=?", (cur.lastrowid,)).fetchone()
    return _as_dict(row)


def list_reminders(user_id: str, include_finished: bool = False) -> list[dict]:
    statuses = "('pending_confirmation', 'active', 'notified')" if not include_finished else "('pending_confirmation', 'active', 'notified', 'completed', 'cancelled')"
    with _conn() as db:
        rows = db.execute(
            f"SELECT * FROM reminders WHERE user_id=? AND status IN {statuses} ORDER BY due_at ASC LIMIT 50",
            (user_id,),
        ).fetchall()
    return [_as_dict(row) for row in rows]


def _latest(user_id: str, statuses: tuple[str, ...]) -> dict | None:
    placeholders = ",".join("?" for _ in statuses)
    with _conn() as db:
        row = db.execute(
            f"SELECT * FROM reminders WHERE user_id=? AND status IN ({placeholders}) ORDER BY id DESC LIMIT 1",
            (user_id, *statuses),
        ).fetchone()
    return _as_dict(row) if row else None


def confirm_latest(user_id: str) -> dict | None:
    reminder = _latest(user_id, ("pending_confirmation",))
    if not reminder:
        return None
    now = _iso(_now())
    with _conn() as db:
        db.execute("UPDATE reminders SET status='active', updated_at=? WHERE id=?", (now, reminder["id"]))
    reminder["status"] = "active"
    return reminder


def cancel_latest(user_id: str) -> dict | None:
    reminder = _latest(user_id, ("pending_confirmation", "active", "notified"))
    if not reminder:
        return None
    with _conn() as db:
        db.execute("UPDATE reminders SET status='cancelled', updated_at=? WHERE id=?", (_iso(_now()), reminder["id"]))
    reminder["status"] = "cancelled"
    return reminder


def complete_latest(user_id: str) -> dict | None:
    reminder = _latest(user_id, ("notified", "active"))
    if not reminder:
        return None
    now = _iso(_now())
    if reminder["repeat_rule"] == "daily":
        next_due = datetime.fromisoformat(reminder["due_at"]).astimezone(TZ) + timedelta(days=1)
        with _conn() as db:
            db.execute(
                "UPDATE reminders SET status='active', due_at=?, notified_at=NULL, completed_at=?, updated_at=? WHERE id=?",
                (_iso(next_due), now, now, reminder["id"]),
            )
        reminder.update(status="active", due_at=_iso(next_due), notified_at=None)
        return reminder
    with _conn() as db:
        db.execute("UPDATE reminders SET status='completed', completed_at=?, updated_at=? WHERE id=?", (now, now, reminder["id"]))
    reminder["status"] = "completed"
    return reminder


def snooze_latest(user_id: str, minutes: int) -> dict | None:
    reminder = _latest(user_id, ("notified", "active"))
    if not reminder:
        return None
    due_at = _now() + timedelta(minutes=minutes)
    with _conn() as db:
        db.execute(
            "UPDATE reminders SET status='active', due_at=?, notified_at=NULL, updated_at=? WHERE id=?",
            (_iso(due_at), _iso(_now()), reminder["id"]),
        )
    reminder.update(status="active", due_at=_iso(due_at), notified_at=None)
    return reminder


def snooze_reminder(user_id: str, reminder_id: int, minutes: int) -> dict | None:
    reminder = get_reminder(user_id, reminder_id)
    if not reminder or reminder["status"] not in ("active", "notified"):
        return None
    due_at = _now() + timedelta(minutes=minutes)
    with _conn() as db:
        db.execute(
            "UPDATE reminders SET status='active', due_at=?, notified_at=NULL, updated_at=? WHERE id=? AND user_id=?",
            (_iso(due_at), _iso(_now()), reminder_id, user_id),
        )
    reminder.update(status="active", due_at=_iso(due_at), notified_at=None)
    return reminder


def complete_reminder(user_id: str, reminder_id: int) -> dict | None:
    reminder = get_reminder(user_id, reminder_id)
    if not reminder or reminder["status"] not in ("active", "notified"):
        return None
    now = _iso(_now())
    if reminder["repeat_rule"] == "daily":
        next_due = datetime.fromisoformat(reminder["due_at"]).astimezone(TZ) + timedelta(days=1)
        with _conn() as db:
            db.execute(
                "UPDATE reminders SET status='active', due_at=?, notified_at=NULL, completed_at=?, updated_at=? WHERE id=? AND user_id=?",
                (_iso(next_due), now, now, reminder_id, user_id),
            )
        reminder.update(status="active", due_at=_iso(next_due), notified_at=None)
        return reminder
    with _conn() as db:
        db.execute(
            "UPDATE reminders SET status='completed', completed_at=?, updated_at=? WHERE id=? AND user_id=?",
            (now, now, reminder_id, user_id),
        )
    reminder["status"] = "completed"
    return reminder


def mark_due_reminders() -> int:
    now = _iso(_now())
    with _conn() as db:
        cur = db.execute(
            "UPDATE reminders SET status='notified', notified_at=?, updated_at=? WHERE status='active' AND due_at<=?",
            (now, now, now),
        )
    return cur.rowcount


def notified_reminders(user_id: str) -> list[dict]:
    with _conn() as db:
        rows = db.execute(
            "SELECT * FROM reminders WHERE user_id=? AND status='notified' ORDER BY due_at ASC", (user_id,)
        ).fetchall()
    return [_as_dict(row) for row in rows]


def get_reminder(user_id: str, reminder_id: int) -> dict | None:
    with _conn() as db:
        row = db.execute("SELECT * FROM reminders WHERE user_id=? AND id=?", (user_id, reminder_id)).fetchone()
    return _as_dict(row) if row else None


def activate_reminder(user_id: str, reminder_id: int) -> dict | None:
    reminder = get_reminder(user_id, reminder_id)
    if not reminder or reminder["status"] != "pending_confirmation":
        return None
    with _conn() as db:
        db.execute("UPDATE reminders SET status='active', updated_at=? WHERE id=? AND user_id=?", (_iso(_now()), reminder_id, user_id))
    reminder["status"] = "active"
    return reminder


def cancel_reminder(user_id: str, reminder_id: int) -> bool:
    with _conn() as db:
        cur = db.execute(
            "UPDATE reminders SET status='cancelled', updated_at=? WHERE id=? AND user_id=? AND status IN ('pending_confirmation','active','notified')",
            (_iso(_now()), reminder_id, user_id),
        )
    return cur.rowcount > 0


def parse_reminder_time(text: str) -> tuple[datetime | None, str]:
    """Parse the deliberately small, predictable first set of Chinese time expressions."""
    now = _now()
    repeat_rule = "daily" if any(x in text for x in ("每天", "每日", "每晚", "每早")) else ""
    relative = re.search(r"([一二两俩三四五六七八九十\d]{1,3})\s*分钟后", text)
    if relative:
        raw_minutes = relative.group(1)
        chinese_numbers = {"一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4, "五": 5,
                           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if raw_minutes.isdigit():
            minutes = int(raw_minutes)
        elif raw_minutes == "十":
            minutes = 10
        elif len(raw_minutes) == 2 and raw_minutes[0] == "十":
            minutes = 10 + chinese_numbers.get(raw_minutes[1], 0)
        elif len(raw_minutes) == 2 and raw_minutes[1] == "十":
            minutes = chinese_numbers.get(raw_minutes[0], 0) * 10
        else:
            minutes = chinese_numbers.get(raw_minutes, 0)
        if not 1 <= minutes <= 180:
            return None, repeat_rule
        return now + timedelta(minutes=minutes), repeat_rule
    match = re.search(r"(?:(明天|明早|明晚|今天|今晚|早上|上午|中午|下午|晚上)\s*)?(\d{1,2})(?:点|时)(?:(\d{1,2})分?)?", text)
    if not match:
        return None, repeat_rule
    period, hour_text, minute_text = match.groups()
    hour, minute = int(hour_text), int(minute_text or 0)
    if hour > 23 or minute > 59:
        return None, repeat_rule
    if period in ("下午", "晚上", "明晚", "今晚") and hour < 12:
        hour += 12
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if period in ("明天", "明早", "明晚"):
        candidate += timedelta(days=1)
    elif period not in ("今天", "今晚") and candidate <= now and not repeat_rule:
        candidate += timedelta(days=1)
    return candidate, repeat_rule


def format_due_at(reminder: dict) -> str:
    due = datetime.fromisoformat(reminder["due_at"]).astimezone(TZ)
    return due.strftime("%m月%d日 %H:%M")
