"""
Integration tests for conversation/history.py

Uses a real SQLite database in a temporary file so tests exercise the actual
SQL queries. Each test class gets its own DB path for isolation.
"""

import os
import tempfile
import pytest

import conversation.history as history_module
from conversation.history import add_message, get_history, clear_history


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """
    Point history.py at a fresh SQLite file for each test.
    Restores the original path afterwards.
    """
    db_path = str(tmp_path / "test_conversations.db")
    original_sqlite = history_module.SQLITE_PATH
    original_data = history_module.DATA_DIR
    history_module.SQLITE_PATH = db_path
    history_module.DATA_DIR = str(tmp_path)
    yield db_path
    history_module.SQLITE_PATH = original_sqlite
    history_module.DATA_DIR = original_data


class TestAddMessage:
    def test_add_user_message(self):
        add_message("conv-1", "user", "Hello!")
        history = get_history("conv-1")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    def test_add_assistant_message(self):
        add_message("conv-1", "assistant", "Hi there!")
        history = get_history("conv-1")
        assert history[0]["role"] == "assistant"

    def test_multiple_messages_stored_in_order(self):
        add_message("conv-2", "user", "First")
        add_message("conv-2", "assistant", "Second")
        add_message("conv-2", "user", "Third")
        history = get_history("conv-2")
        assert [m["content"] for m in history] == ["First", "Second", "Third"]

    def test_different_conversations_are_isolated(self):
        add_message("conv-A", "user", "Message for A")
        add_message("conv-B", "user", "Message for B")
        assert get_history("conv-A")[0]["content"] == "Message for A"
        assert get_history("conv-B")[0]["content"] == "Message for B"


class TestGetHistory:
    def test_empty_conversation_returns_empty_list(self):
        assert get_history("no-such-conv") == []

    def test_returns_oldest_first(self):
        add_message("conv-1", "user", "first")
        add_message("conv-1", "assistant", "second")
        history = get_history("conv-1")
        assert history[0]["content"] == "first"
        assert history[1]["content"] == "second"

    def test_respects_limit(self):
        for i in range(15):
            add_message("conv-1", "user", f"msg {i}")
        history = get_history("conv-1", limit=5)
        assert len(history) == 5

    def test_limit_returns_most_recent(self):
        for i in range(5):
            add_message("conv-1", "user", f"msg {i}")
        history = get_history("conv-1", limit=2)
        # Should be the last 2 messages
        assert history[0]["content"] == "msg 3"
        assert history[1]["content"] == "msg 4"

    def test_returned_format_matches_openai_spec(self):
        add_message("conv-1", "user", "test")
        msg = get_history("conv-1")[0]
        assert set(msg.keys()) == {"role", "content"}


class TestClearHistory:
    def test_clear_removes_all_messages(self):
        add_message("conv-1", "user", "to be deleted")
        clear_history("conv-1")
        assert get_history("conv-1") == []

    def test_clear_returns_deleted_count(self):
        add_message("conv-1", "user", "one")
        add_message("conv-1", "assistant", "two")
        deleted = clear_history("conv-1")
        assert deleted == 2

    def test_clear_nonexistent_conversation_returns_zero(self):
        assert clear_history("ghost-conv") == 0

    def test_clear_only_affects_target_conversation(self):
        add_message("conv-keep", "user", "keep me")
        add_message("conv-del", "user", "delete me")
        clear_history("conv-del")
        assert len(get_history("conv-keep")) == 1
        assert get_history("conv-del") == []
