"""
In-memory TTL (time-to-live) cache.

Two instances are created at module level:
  embedding_cache  — caches text → vector results for 24h.
                     Embeddings are deterministic so the same text always
                     produces the same vector — safe to cache indefinitely.
  routing_cache    — caches question → agent name for 5 min.
                     Short TTL because the agent registry can change.
"""

import time
import hashlib
from typing import Any
from config import EMBEDDING_CACHE_TTL, ROUTING_CACHE_TTL


class TTLCache:
    """
    A simple dict-backed cache where each entry expires after `ttl_seconds`.

    Keys are derived by hashing all positional arguments passed to get()/set(),
    so callers can use any hashable values as a compound key.

    Thread safety: Python's GIL makes individual dict operations atomic,
    so this is safe for single-process use without explicit locking.
    """

    def __init__(self, name: str, ttl_seconds: int):
        self.name = name
        self.ttl = ttl_seconds
        # Stores {hashed_key: (value, expiry_timestamp)}
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, *args) -> str:
        """Hash all args into a single fixed-length cache key."""
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args) -> Any | None:
        """
        Return the cached value for the given args, or None if missing/expired.

        On expiry the stale entry is evicted immediately (lazy deletion —
        we don't run a background cleanup job).
        """
        key = self._key(*args)
        entry = self._store.get(key)
        if entry:
            value, expires_at = entry
            if time.time() < expires_at:
                self._hits += 1
                return value
            del self._store[key]  # evict expired entry
        self._misses += 1
        return None

    def set(self, *args, value: Any) -> None:
        """
        Store a value under the key derived from *args.

        value must be passed as a keyword argument to distinguish it from
        the key components in *args.
        """
        key = self._key(*args)
        self._store[key] = (value, time.time() + self.ttl)

    def stats(self) -> dict:
        """Return hit/miss counts, hit rate, and current cache size."""
        total = self._hits + self._misses
        rate = self._hits / total if total else 0.0
        return {
            "name": self.name,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(rate, 3),
            "size": len(self._store),
        }

    def clear(self) -> None:
        """Evict all entries (used in tests and the /cache/clear endpoint)."""
        self._store.clear()


# Module-level singletons shared across all imports
embedding_cache = TTLCache("embeddings", ttl_seconds=EMBEDDING_CACHE_TTL)
routing_cache = TTLCache("routing", ttl_seconds=ROUTING_CACHE_TTL)
