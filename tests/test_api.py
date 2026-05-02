"""
Functional tests for the FastAPI endpoints in main.py.

All external dependencies (OpenAI, ChromaDB queries, SQLite reads) are mocked
so these tests exercise only the HTTP layer and endpoint logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


VALID_KEY = "test-api-key"
HEADERS = {"X-API-Key": VALID_KEY}

SAMPLE_AGENT = {
    "name": "python-agent",
    "description": "Python expert",
    "system_prompt": "You are a Python expert.",
    "collection": "python_docs",
    "keywords": ["python"],
}


@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with the full pipeline mocked out.
    scope=module so the app is only created once per test file.
    """
    with (
        patch("main._seed_knowledge_base"),
        patch("main.select_agent", return_value=(SAMPLE_AGENT, "keyword")),
        patch("main.retrieve", return_value=["Python uses indentation."]),
        patch("main.search_memories", return_value=[]),
        patch("main.stream_answer", return_value=iter(["Hello", " from", " Python-agent"])),
        patch("main.maybe_extract_and_store"),
    ):
        from main import app
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_returns_ok_status(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_no_auth_required(self, client):
        # Should work without X-API-Key
        assert client.get("/health").status_code == 200


# ── Auth middleware ───────────────────────────────────────────────────────────

class TestAuth:
    def test_missing_api_key_returns_401(self, client):
        resp = client.post("/chat", json={"question": "hi"})
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client):
        resp = client.post(
            "/chat",
            json={"question": "hi"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_correct_api_key_passes(self, client):
        with patch("main.stream_answer", return_value=iter(["ok"])):
            resp = client.post("/chat", json={"question": "hi"}, headers=HEADERS)
        assert resp.status_code == 200


# ── POST /chat ────────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_returns_200(self, client):
        with patch("main.stream_answer", return_value=iter(["answer"])):
            resp = client.post("/chat", json={"question": "What is Python?"}, headers=HEADERS)
        assert resp.status_code == 200

    def test_response_contains_answer(self, client):
        with patch("main.stream_answer", return_value=iter(["Hello", " world"])):
            resp = client.post("/chat", json={"question": "hi"}, headers=HEADERS)
        body = resp.json()
        assert body["answer"] == "Hello world"

    def test_response_contains_agent_used(self, client):
        with patch("main.stream_answer", return_value=iter(["ans"])):
            body = client.post("/chat", json={"question": "hi"}, headers=HEADERS).json()
        assert body["agent_used"] == "python-agent"

    def test_response_contains_conversation_id(self, client):
        with patch("main.stream_answer", return_value=iter(["ans"])):
            body = client.post("/chat", json={"question": "hi"}, headers=HEADERS).json()
        assert "conversation_id" in body
        assert len(body["conversation_id"]) > 0

    def test_provided_conversation_id_preserved(self, client):
        with patch("main.stream_answer", return_value=iter(["ans"])):
            body = client.post(
                "/chat",
                json={"question": "hi", "conversation_id": "my-conv-123"},
                headers=HEADERS,
            ).json()
        assert body["conversation_id"] == "my-conv-123"

    def test_facts_retrieved_count(self, client):
        with (
            patch("main.retrieve", return_value=["fact1", "fact2"]),
            patch("main.stream_answer", return_value=iter(["ans"])),
        ):
            body = client.post("/chat", json={"question": "hi"}, headers=HEADERS).json()
        assert body["facts_retrieved"] == 2

    def test_match_type_in_response(self, client):
        with patch("main.stream_answer", return_value=iter(["ans"])):
            body = client.post("/chat", json={"question": "hi"}, headers=HEADERS).json()
        assert "match_type" in body


# ── POST /chat/stream ─────────────────────────────────────────────────────────

class TestChatStreamEndpoint:
    def test_returns_200(self, client):
        with patch("main.stream_answer", return_value=iter(["token"])):
            resp = client.post(
                "/chat/stream", json={"question": "hi"}, headers=HEADERS
            )
        assert resp.status_code == 200

    def test_content_type_is_event_stream(self, client):
        with patch("main.stream_answer", return_value=iter(["token"])):
            resp = client.post(
                "/chat/stream", json={"question": "hi"}, headers=HEADERS
            )
        assert "text/event-stream" in resp.headers["content-type"]

    def test_response_contains_sse_data_lines(self, client):
        with patch("main.stream_answer", return_value=iter(["hello"])):
            resp = client.post(
                "/chat/stream", json={"question": "hi"}, headers=HEADERS
            )
        assert "data:" in resp.text

    def test_stream_ends_with_done(self, client):
        with patch("main.stream_answer", return_value=iter(["hi"])):
            resp = client.post(
                "/chat/stream", json={"question": "hi"}, headers=HEADERS
            )
        assert "[DONE]" in resp.text

    def test_agent_name_in_stream(self, client):
        with patch("main.stream_answer", return_value=iter(["hi"])):
            resp = client.post(
                "/chat/stream", json={"question": "hi"}, headers=HEADERS
            )
        assert "python-agent" in resp.text


# ── POST /ingest ──────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_returns_200(self, client):
        with patch("main.ingest_text", return_value=3):
            resp = client.post(
                "/ingest",
                json={"collection": "python_docs", "text": "hello world", "source": "test"},
                headers=HEADERS,
            )
        assert resp.status_code == 200

    def test_response_contains_chunk_count(self, client):
        with patch("main.ingest_text", return_value=5):
            body = client.post(
                "/ingest",
                json={"collection": "python_docs", "text": "some text", "source": "api"},
                headers=HEADERS,
            ).json()
        assert body["chunks_added"] == 5

    def test_response_contains_collection_name(self, client):
        with patch("main.ingest_text", return_value=1):
            body = client.post(
                "/ingest",
                json={"collection": "devops_docs", "text": "some text"},
                headers=HEADERS,
            ).json()
        assert body["collection"] == "devops_docs"

    def test_requires_auth(self, client):
        resp = client.post(
            "/ingest",
            json={"collection": "col", "text": "text"},
        )
        assert resp.status_code == 401


# ── DELETE /conversation/{id} ─────────────────────────────────────────────────

class TestDeleteConversationEndpoint:
    def test_returns_200(self, client):
        with patch("main.clear_history", return_value=2):
            resp = client.delete("/conversation/abc-123", headers=HEADERS)
        assert resp.status_code == 200

    def test_returns_deleted_count(self, client):
        with patch("main.clear_history", return_value=4):
            body = client.delete("/conversation/abc-123", headers=HEADERS).json()
        assert body["messages_deleted"] == 4

    def test_returns_conversation_id(self, client):
        with patch("main.clear_history", return_value=0):
            body = client.delete("/conversation/my-conv", headers=HEADERS).json()
        assert body["conversation_id"] == "my-conv"

    def test_requires_auth(self, client):
        resp = client.delete("/conversation/abc")
        assert resp.status_code == 401


# ── GET /cache/stats ──────────────────────────────────────────────────────────

class TestCacheStatsEndpoint:
    def test_returns_200(self, client):
        assert client.get("/cache/stats", headers=HEADERS).status_code == 200

    def test_response_has_embedding_and_routing_keys(self, client):
        body = client.get("/cache/stats", headers=HEADERS).json()
        assert "embedding_cache" in body
        assert "routing_cache" in body

    def test_stats_have_expected_fields(self, client):
        body = client.get("/cache/stats", headers=HEADERS).json()
        for cache_key in ("embedding_cache", "routing_cache"):
            stats = body[cache_key]
            assert "hits" in stats
            assert "misses" in stats
            assert "hit_rate" in stats

    def test_requires_auth(self, client):
        resp = client.get("/cache/stats")
        assert resp.status_code == 401
