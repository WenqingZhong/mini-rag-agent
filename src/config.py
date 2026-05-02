"""
Central configuration — reads from environment variables with sensible defaults.
Production equivalent: @Value("${...}") annotations scattered across Spring services.

By centralising here, every module imports from config.py instead of calling
os.getenv() individually — easier to audit and change in one place.
"""

import os

# ── Models ───────────────────────────────────────────────────────────────────
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
ROUTING_MODEL = os.getenv("ROUTING_MODEL", "gpt-4o-mini")

# ── Token Budgets ─────────────────────────────────────────────────────────────
# Total tokens we target per request (gpt-4o-mini supports 128k, but a tighter
# budget keeps latency and cost predictable).
# Production: TokenManagementService uses JTokkitTokenCountEstimator per model.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))
BUDGET_SYSTEM_PROMPT = int(os.getenv("BUDGET_SYSTEM_PROMPT", "1500"))
BUDGET_FACTS = int(os.getenv("BUDGET_FACTS", "3000"))
BUDGET_MEMORY = int(os.getenv("BUDGET_MEMORY", "500"))
BUDGET_HISTORY = int(os.getenv("BUDGET_HISTORY", "4000"))
# Remaining tokens (~3000) become the LLM's answer budget via max_tokens

# ── RAG Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "150"))  # ~200 tokens
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "20"))
TOP_K_FACTS = int(os.getenv("TOP_K_FACTS", "3"))

# ── Storage Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/conversations.db")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory.db")

# ── Auth ──────────────────────────────────────────────────────────────────────
# Set API_KEY env var to a real secret in production.
# Production: Spring Security validates JWT tokens from PingFederate.
API_KEY = os.getenv("API_KEY", "dev-secret-key-change-me")

# ── Cache TTL (seconds) ───────────────────────────────────────────────────────
# Embeddings are deterministic — safe to cache for a long TTL.
# Routing decisions expire faster because the agent registry can change.
# Production: CaffeineCacheConfig sets per-cache TTL via expireAfterWrite().
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))  # 24 h
ROUTING_CACHE_TTL = int(os.getenv("ROUTING_CACHE_TTL", "300"))  # 5 min
