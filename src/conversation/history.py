"""
CONCEPT: Conversation History (TIER 1 — SQLite Persistence)
============================================================
Production equivalent: MessageManagementService + MessageRepository (Cosmos DB)

Previously: in-memory dict — wiped on every server restart.
Now:        SQLite database at ./data/conversations.db

SCHEMA:
    messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role            TEXT NOT NULL,   -- "user" or "assistant"
        content         TEXT NOT NULL,
        created_at      TEXT DEFAULT CURRENT_TIMESTAMP
    )
    Index on conversation_id for O(log n) lookups.

WHY SQLite OVER IN-MEMORY?
  - Survives restarts  → chat history is preserved
  - Easy to inspect    → open with TablePlus, DBeaver, etc.
  - Zero infrastructure → no Redis, no Cosmos, no setup needed
  - Same SQL API       → easy to swap for PostgreSQL later

PRODUCTION DIFFERENCE:
  Cosmos DB scales horizontally; SQLite is a single-file embedded DB.
  The public API (add_message / get_history / clear_history) is identical —
  only the storage backend differs.
"""

import sqlite3
import os
from utils.logger import get_logger
from config import SQLITE_PATH, DATA_DIR

logger = get_logger("history")


def _get_conn() -> sqlite3.Connection:
    """Open a SQLite connection and ensure the schema exists."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON messages(conversation_id)")
    conn.commit()
    return conn


def add_message(conversation_id: str, role: str, content: str) -> None:
    """
    Persist one message to SQLite.
    Production equivalent: MessageManagementService.createMessage()
    """
    conn = _get_conn()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content),
    )
    conn.commit()
    conn.close()
    logger.debug(
        "Message saved",
        extra={
            "conversation_id": conversation_id,
            "role": role,
            "preview": content[:60],
        },
    )


def get_history(conversation_id: str, limit: int = 10) -> list[dict]:
    """
    Return the last `limit` messages for a conversation, oldest first.
    The returned dicts match OpenAI's message format and can be passed
    directly to provider.py without transformation.
    Production equivalent: MessageManagementService.getSimpleHistory()
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT role, content FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (conversation_id, limit),
    ).fetchall()
    conn.close()
    # Rows come back newest-first (DESC); reverse to chronological order
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def clear_history(conversation_id: str) -> int:
    """
    Delete all messages for a conversation.
    Returns the number of rows deleted.
    Production equivalent: ConversationManagementService.deleteConversation()
    """
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    logger.info(
        "History cleared",
        extra={"conversation_id": conversation_id, "deleted": deleted},
    )
    return deleted


def conversation_count() -> int:
    """Return the number of distinct conversation IDs. (Debug/admin utility)"""
    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(DISTINCT conversation_id) FROM messages"
    ).fetchone()[0]
    conn.close()
    return count
