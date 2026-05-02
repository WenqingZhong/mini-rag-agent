"""
Unit tests for utils/cache.py

Tests TTLCache behaviour: get/set, expiry, stats, clear, and compound keys.
Uses time.time() mocking to control TTL without sleeping.
"""

from unittest.mock import patch
from utils.cache import TTLCache


class TestTTLCacheGetSet:
    def test_set_then_get_returns_value(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("key", value="hello")
        assert cache.get("key") == "hello"

    def test_get_missing_key_returns_none(self):
        cache = TTLCache("test", ttl_seconds=60)
        assert cache.get("missing") is None

    def test_overwrite_key(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("k", value="first")
        cache.set("k", value="second")
        assert cache.get("k") == "second"

    def test_different_keys_stored_independently(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("a", value=1)
        cache.set("b", value=2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    def test_compound_key_from_multiple_args(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("part1", "part2", value="compound")
        assert cache.get("part1", "part2") == "compound"
        assert cache.get("part1") is None  # different key


class TestTTLCacheExpiry:
    def test_expired_entry_returns_none(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("k", value="v")
        # Simulate time has passed beyond TTL
        with patch("utils.cache.time.time", return_value=9_999_999_999):
            assert cache.get("k") is None

    def test_expired_entry_is_evicted(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("k", value="v")
        with patch("utils.cache.time.time", return_value=9_999_999_999):
            cache.get("k")
        assert len(cache._store) == 0

    def test_not_yet_expired_entry_is_returned(self):
        cache = TTLCache("test", ttl_seconds=3600)
        cache.set("k", value="alive")
        assert cache.get("k") == "alive"


class TestTTLCacheStats:
    def test_initial_stats(self):
        cache = TTLCache("mycache", ttl_seconds=60)
        stats = cache.stats()
        assert stats["name"] == "mycache"
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0

    def test_hit_increments_hits(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("k", value="v")
        cache.get("k")
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 0

    def test_miss_increments_misses(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.get("missing")
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"] == 0

    def test_hit_rate_calculation(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("k", value="v")
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("x")  # miss
        stats = cache.stats()
        assert stats["hit_rate"] == round(2 / 3, 3)

    def test_hit_rate_zero_when_no_requests(self):
        cache = TTLCache("test", ttl_seconds=60)
        assert cache.stats()["hit_rate"] == 0.0

    def test_size_reflects_stored_entries(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("a", value=1)
        cache.set("b", value=2)
        assert cache.stats()["size"] == 2


class TestTTLCacheClear:
    def test_clear_removes_all_entries(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("a", value=1)
        cache.set("b", value=2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.stats()["size"] == 0

    def test_clear_does_not_reset_stats(self):
        cache = TTLCache("test", ttl_seconds=60)
        cache.set("k", value="v")
        cache.get("k")  # hit
        cache.clear()
        # Stats persist after clear (counts are not reset)
        assert cache.stats()["hits"] == 1
