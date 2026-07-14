"""Memory store — SQLite for companion profiles and user-controlled facts."""
import sqlite3
import json
import os
from datetime import datetime, timezone
from .config import STATE_DB_PATH


def _conn():
    os.makedirs(os.path.dirname(STATE_DB_PATH), exist_ok=True)
    c = sqlite3.connect(STATE_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    """Create memory tables if they don't exist."""
    with _conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                family TEXT DEFAULT '[]',
                interests TEXT DEFAULT '[]',
                notes TEXT DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                session_id TEXT,
                created_at TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id)")
        # Older deployments created user_facts without session_id. Migrate in
        # place so new explicit memory actions work with existing user data.
        columns = {row["name"] for row in db.execute("PRAGMA table_info(user_facts)").fetchall()}
        if "session_id" not in columns:
            db.execute("ALTER TABLE user_facts ADD COLUMN session_id TEXT")
        if "created_at" not in columns:
            db.execute("ALTER TABLE user_facts ADD COLUMN created_at TEXT")
            # Preserve the historical timestamp where this deployment used
            # the old extracted_at name.
            if "extracted_at" in columns:
                db.execute("UPDATE user_facts SET created_at=extracted_at WHERE created_at IS NULL")
        db.commit()


def get_facts(user_id: str, limit: int = 10) -> list[str]:
    with _conn() as db:
        rows = db.execute(
            "SELECT fact FROM user_facts WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [r["fact"] for r in rows]


def list_facts(user_id: str, limit: int = 30) -> list[dict]:
    """User-facing memory listing, including IDs needed by management APIs."""
    with _conn() as db:
        rows = db.execute(
            "SELECT id, fact, session_id, created_at FROM user_facts WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def add_fact(user_id: str, fact: str, session_id: str = "") -> None:
    fact = fact.strip()
    if not fact:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as db:
        # Facts are extracted after every turn.  Keeping identical records makes
        # the companion repeatedly foreground the same detail in later chats.
        existing = db.execute(
            "SELECT 1 FROM user_facts WHERE user_id=? AND fact=? LIMIT 1",
            (user_id, fact),
        ).fetchone()
        if existing:
            return
        db.execute(
            "INSERT INTO user_facts (user_id, fact, session_id, created_at) VALUES (?,?,?,?)",
            (user_id, fact, session_id, now),
        )
        db.commit()


def delete_fact(user_id: str, fact_id: int) -> bool:
    with _conn() as db:
        cur = db.execute("DELETE FROM user_facts WHERE user_id=? AND id=?", (user_id, fact_id))
        db.commit()
    return cur.rowcount > 0


def delete_facts_matching(user_id: str, phrase: str) -> int:
    """Delete only facts explicitly matched by the user, never broad guesses."""
    phrase = phrase.strip()
    if not phrase:
        return 0
    with _conn() as db:
        cur = db.execute(
            "DELETE FROM user_facts WHERE user_id=? AND fact LIKE ?",
            (user_id, f"%{phrase}%"),
        )
        db.commit()
    return cur.rowcount


def clear_facts(user_id: str) -> int:
    with _conn() as db:
        cur = db.execute("DELETE FROM user_facts WHERE user_id=?", (user_id,))
        db.commit()
    return cur.rowcount


def get_profile(user_id: str) -> dict | None:
    with _conn() as db:
        row = db.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["family"] = json.loads(d.get("family", "[]"))
        d["interests"] = json.loads(d.get("interests", "[]"))
        return d


def upsert_profile(user_id: str, fields: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    existing = get_profile(user_id)
    if existing:
        updates = {**fields}
        for k in ("family", "interests"):
            if k in updates:
                updates[k] = json.dumps(updates[k], ensure_ascii=False)
        updates["updated_at"] = now
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [user_id]
        with _conn() as db:
            db.execute(f"UPDATE user_profiles SET {sets} WHERE user_id=?", vals)
            db.commit()
    else:
        d = {"user_id": user_id, "name": "", "age": None, "family": "[]",
             "interests": "[]", "notes": "", "updated_at": now}
        d.update(fields)
        for k in ("family", "interests"):
            d[k] = json.dumps(d.get(k, "[]"), ensure_ascii=False)
        cols = ", ".join(d.keys())
        ph = ", ".join("?" for _ in d)
        with _conn() as db:
            db.execute(f"INSERT INTO user_profiles ({cols}) VALUES ({ph})", list(d.values()))
            db.commit()


init()
