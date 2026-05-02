"""
Integration tests for conversation/memory.py

SQLite uses a temp file per test. OpenAI embedding calls are mocked with
a deterministic fake vector so cosine similarity is predictable.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

import conversation.memory as memory_module
from conversation.memory import store_memory, search_memories, maybe_extract_and_store


def _fake_vector(seed: float = 1.0) -> list[float]:
    """Return a unit vector along the first dimension — easy to reason about."""
    vec = [0.0] * 1536
    vec[0] = seed
    return vec


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Fresh SQLite file for every test."""
    db_path = str(tmp_path / "test_memory.db")
    original_db = memory_module.MEMORY_DB_PATH
    original_dir = memory_module.DATA_DIR
    memory_module.MEMORY_DB_PATH = db_path
    memory_module.DATA_DIR = str(tmp_path)
    yield
    memory_module.MEMORY_DB_PATH = original_db
    memory_module.DATA_DIR = original_dir


@pytest.fixture
def mock_embed():
    """Patch memory._embed to return a fixed vector without calling OpenAI."""
    with patch.object(memory_module, "_embed", return_value=_fake_vector(1.0)) as m:
        yield m


class TestStoreMemory:
    def test_stored_memory_is_retrievable(self, mock_embed):
        store_memory("user-1", "User prefers Python 3.11")
        memories = search_memories("user-1", "Python version", min_similarity=0.0)
        assert any("Python 3.11" in m for m in memories)

    def test_stores_for_correct_user(self, mock_embed):
        store_memory("user-A", "fact about A")
        store_memory("user-B", "fact about B")
        memories_a = search_memories("user-A", "fact", min_similarity=0.0)
        assert all("A" in m for m in memories_a)

    def test_multiple_memories_stored(self, mock_embed):
        store_memory("user-1", "fact one")
        store_memory("user-1", "fact two")
        memories = search_memories("user-1", "fact", min_similarity=0.0)
        assert len(memories) == 2


class TestSearchMemories:
    def test_empty_store_returns_empty_list(self, mock_embed):
        result = search_memories("new-user", "anything")
        assert result == []

    def test_below_threshold_excluded(self):
        """Memories whose similarity is below min_similarity are filtered out."""
        # Store a memory with vector [1, 0, 0, ...]
        with patch.object(memory_module, "_embed", return_value=_fake_vector(1.0)):
            store_memory("user-1", "relevant fact")

        # Query with orthogonal vector [0, 1, 0, ...] — similarity = 0
        orthogonal = [0.0] * 1536
        orthogonal[1] = 1.0
        with patch.object(memory_module, "_embed", return_value=orthogonal):
            results = search_memories("user-1", "query", min_similarity=0.5)

        assert results == []

    def test_above_threshold_included(self):
        """Memories whose similarity is above threshold are returned."""
        vec = _fake_vector(1.0)
        with patch.object(memory_module, "_embed", return_value=vec):
            store_memory("user-1", "high similarity fact")
            results = search_memories("user-1", "query", min_similarity=0.99)

        assert len(results) == 1

    def test_respects_top_k(self, mock_embed):
        for i in range(5):
            store_memory("user-1", f"memory {i}")
        results = search_memories("user-1", "memory", top_k=2, min_similarity=0.0)
        assert len(results) <= 2


class TestMaybeExtractAndStore:
    def test_skips_short_conversation(self, mock_embed):
        # Fewer than 2 messages — nothing to extract
        maybe_extract_and_store("user-1", [{"role": "user", "content": "hi"}])
        assert search_memories("user-1", "anything", min_similarity=0.0) == []

    def test_extracts_and_stores_facts(self, mock_embed):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"facts": ["User likes Python", "User is a beginner"]}
        )
        conversation = [
            {"role": "user", "content": "I am new to Python"},
            {"role": "assistant", "content": "Welcome! Python is great for beginners."},
        ]
        with patch.object(memory_module.client.chat.completions, "create", return_value=mock_response):
            maybe_extract_and_store("user-1", conversation)

        memories = search_memories("user-1", "Python", min_similarity=0.0)
        assert len(memories) == 2

    def test_no_facts_stores_nothing(self, mock_embed):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"facts": []})
        conversation = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        with patch.object(memory_module.client.chat.completions, "create", return_value=mock_response):
            maybe_extract_and_store("user-1", conversation)

        assert search_memories("user-1", "hello", min_similarity=0.0) == []

    def test_llm_failure_does_not_raise(self, mock_embed):
        with patch.object(
            memory_module.client.chat.completions,
            "create",
            side_effect=Exception("API down"),
        ):
            # Should log a warning and not propagate the exception
            maybe_extract_and_store(
                "user-1",
                [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "ok"},
                ],
            )
