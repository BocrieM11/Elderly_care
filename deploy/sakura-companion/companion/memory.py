"""Memory store — SQLite for user profiles and facts. No vector DB, no embeddings."""
import sqlite3
import json
from datetime import datetime, timezone
from .config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
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
        db.commit()


def get_facts(user_id: str, limit: int = 10) -> list[str]:
    """Get recent facts about a user."""
    with _conn() as db:
        rows = db.execute(
            "SELECT fact FROM user_facts WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [r["fact"] for r in rows]


def add_fact(user_id: str, fact: str, session_id: str = "") -> None:
    """Store a new fact."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as db:
        db.execute(
            "INSERT INTO user_facts (user_id, fact, session_id, created_at) VALUES (?,?,?,?)",
            (user_id, fact, session_id, now),
        )
        db.commit()


def get_profile(user_id: str) -> dict | None:
    """Get user profile."""
    with _conn() as db:
        row = db.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["family"] = json.loads(d.get("family", "[]"))
        d["interests"] = json.loads(d.get("interests", "[]"))
        return d


def upsert_profile(user_id: str, fields: dict) -> None:
    """Create or update user profile."""
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


# Auto-init on import
init()
