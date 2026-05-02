"""
TIER 2: Long-Term User Memory
================================
Production equivalent: MemoryService.searchMemories() + maybeStoreMemory()

WHAT IS MEMORY VS HISTORY?
  History   = short-term, per-session messages (wiped when session ends or trimmed).
  Memory    = long-term, cross-session facts ABOUT the user (persists forever).

  Example memories:
    "User prefers Python 3.11"
    "User's project uses FastAPI"
    "User has trouble understanding async/await"

  These are recalled in future sessions to personalise answers — even in brand
  new conversations the agent "remembers" the user.

HOW IT WORKS:
  1. STORE: After each response, _extract_facts() asks the LLM to pull 0-3
     memorable facts from the last conversation turn. Each fact is embedded
     and stored in SQLite alongside its vector.
     Production: MemoryService.maybeStoreMemory() → external memory microservice.

  2. SEARCH: At the start of each request, search_memories() embeds the
     current question and finds stored facts with cosine similarity > threshold.
     Production: MemoryService.searchMemories() → POST /memories/search
     to the external service.

  3. INJECT: Relevant memories are passed to provider.py and injected into
     the system prompt as "WHAT I KNOW ABOUT YOU" (see provider.py).
     Production: results of searchMemories() injected in StreamingChatService.

STORAGE:
  SQLite table: memories (user_id, content, embedding JSON, created_at)
  Cosine similarity is computed in Python — fast enough for <10k memories.
  For larger scale: use ChromaDB with a "memories" collection, or Pinecone.
"""

import sqlite3
import json
import math
import os
from openai import OpenAI
from utils.logger import get_logger
from config import MEMORY_DB_PATH, DATA_DIR, EMBEDDING_MODEL

logger = get_logger("memory")
client = OpenAI()


# ── Storage helpers ───────────────────────────────────────────────────────────


def _get_conn() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            embedding  TEXT    NOT NULL,
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)")
    conn.commit()
    return conn


def _embed(text: str) -> list[float]:
    """Embed text for memory storage/search. Uses the same model as retriever."""
    resp = client.embeddings.create(
        input=text.replace("\n", " ").strip(),
        model=EMBEDDING_MODEL,
    )
    return resp.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / mag if mag else 0.0


# ── Public API ────────────────────────────────────────────────────────────────


def store_memory(user_id: str, content: str) -> None:
    """
    Embed and persist a single memory fact for a user.

    Production equivalent: MemoryService.maybeStoreMemory() → POST /memories
    """
    vector = _embed(content)
    conn = _get_conn()
    conn.execute(
        "INSERT INTO memories (user_id, content, embedding) VALUES (?, ?, ?)",
        (user_id, content, json.dumps(vector)),
    )
    conn.commit()
    conn.close()
    logger.info("Memory stored", extra={"user_id": user_id, "content": content[:60]})


def search_memories(
    user_id: str,
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.75,
) -> list[str]:
    """
    Retrieve memories relevant to the current query using cosine similarity.

    Returns a list of memory strings (injected into the system prompt).
    Only returns memories above min_similarity to filter out irrelevant noise.

    Production equivalent: MemoryService.searchMemories() → POST /memories/search
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT content, embedding FROM memories WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    query_vec = _embed(query)
    scored = [
        (row[0], _cosine_similarity(query_vec, json.loads(row[1]))) for row in rows
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = [content for content, score in scored[:top_k] if score >= min_similarity]
    logger.info(
        "Memories searched",
        extra={
            "user_id": user_id,
            "total": len(rows),
            "returned": len(results),
            "threshold": min_similarity,
        },
    )
    return results


def maybe_extract_and_store(
    user_id: str,
    conversation: list[dict],
) -> None:
    """
    Ask the LLM to extract memorable facts from the conversation, then store them.

    Called asynchronously AFTER each response so it doesn't block streaming.
    Production: MemoryService.maybeStoreMemory() sends conversation to an
    external mem0 service that handles extraction automatically.

    Args:
        user_id:      The current user's identifier
        conversation: Recent messages [{"role": ..., "content": ...}]
    """
    if not conversation or len(conversation) < 2:
        return

    # Only look at the last 2 turns (1 user + 1 assistant exchange)
    recent = conversation[-4:] if len(conversation) >= 4 else conversation

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze conversations and extract memorable facts about the USER. "
                        "Focus on: preferences, technical setup, recurring issues, explicit requests. "
                        'Return a JSON object: {"facts": ["...", "..."]}. '
                        'Return {"facts": []} if nothing notable is worth remembering. '
                        "Maximum 3 facts. Be concise (one sentence each)."
                    ),
                },
                {"role": "user", "content": json.dumps(recent)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        data = json.loads(resp.choices[0].message.content)
        facts = data.get("facts", [])

        for fact in facts:
            if fact.strip():
                store_memory(user_id, fact.strip())

        if facts:
            logger.info(
                "Memories extracted and stored",
                extra={"user_id": user_id, "count": len(facts)},
            )

    except Exception as e:
        # Memory extraction is non-critical — log and continue
        logger.warning("Memory extraction failed", extra={"error": str(e)})
