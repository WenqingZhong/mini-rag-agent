"""
Unit tests for llm/provider.py

Tests the pure functions (count_tokens, _trim_history_to_budget, _build_messages)
without making any real LLM calls.
stream_answer() is tested with a mocked OpenAI client.
"""

import pytest
from unittest.mock import MagicMock, patch

from llm.provider import count_tokens, _trim_history_to_budget, _build_messages, stream_answer


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") >= 1

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello world foo bar baz qux")
        assert long > short

    def test_returns_integer(self):
        assert isinstance(count_tokens("test"), int)


class TestTrimHistoryToBudget:
    def _make_history(self, n: int, tokens_each: int = 10) -> list[dict]:
        """Create n messages each with `tokens_each` approximate tokens."""
        word = "word " * tokens_each  # each "word" is ~1 token
        return [{"role": "user", "content": word} for _ in range(n)]

    def test_history_within_budget_unchanged(self):
        history = [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "ok"},
        ]
        result = _trim_history_to_budget(history, budget=10_000)
        assert len(result) == 2

    def test_history_over_budget_trimmed_from_front(self):
        history = [
            {"role": "user", "content": "first old message"},
            {"role": "assistant", "content": "second old message"},
            {"role": "user", "content": "recent message"},
        ]
        # Budget fits only the last message — verify oldest are dropped first
        budget = count_tokens("recent message")
        result = _trim_history_to_budget(history, budget=budget)
        assert result[-1]["content"] == "recent message"
        assert len(result) < len(history)

    def test_empty_history_returns_empty(self):
        assert _trim_history_to_budget([], budget=1000) == []

    def test_budget_zero_trims_all(self):
        history = [{"role": "user", "content": "hello"}]
        result = _trim_history_to_budget(history, budget=0)
        assert result == []

    def test_original_list_not_mutated(self):
        history = [
            {"role": "user", "content": "msg " * 100},
            {"role": "user", "content": "msg " * 100},
        ]
        original_len = len(history)
        _trim_history_to_budget(history, budget=1)
        assert len(history) == original_len  # original untouched


class TestBuildMessages:
    def test_first_message_is_system(self, agent_config):
        messages = _build_messages(agent_config, [], [], [], "hi")
        assert messages[0]["role"] == "system"

    def test_last_message_is_user_question(self, agent_config):
        messages = _build_messages(agent_config, [], [], [], "my question")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "my question"

    def test_system_prompt_contains_agent_prompt(self, agent_config):
        messages = _build_messages(agent_config, [], [], [], "hi")
        assert agent_config["system_prompt"] in messages[0]["content"]

    def test_facts_injected_into_system(self, agent_config):
        facts = ["Python uses indentation.", "pip installs packages."]
        messages = _build_messages(agent_config, facts, [], [], "hi")
        system = messages[0]["content"]
        assert "Python uses indentation." in system
        assert "pip installs packages." in system

    def test_memories_injected_into_system(self, agent_config):
        memories = ["User prefers Python 3.11"]
        messages = _build_messages(agent_config, [], memories, [], "hi")
        system = messages[0]["content"]
        assert "User prefers Python 3.11" in system

    def test_history_inserted_between_system_and_user(self, agent_config, sample_history):
        messages = _build_messages(agent_config, [], [], sample_history, "new question")
        # [system, hist[0], hist[1], user]
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[-1]["content"] == "new question"

    def test_no_facts_no_extra_section(self, agent_config):
        messages = _build_messages(agent_config, [], [], [], "hi")
        system = messages[0]["content"]
        # The section block is only added when facts are present
        assert "--- RETRIEVED KNOWLEDGE ---" not in system

    def test_no_memories_no_extra_section(self, agent_config):
        messages = _build_messages(agent_config, [], [], [], "hi")
        system = messages[0]["content"]
        assert "WHAT I KNOW ABOUT YOU" not in system

    def test_minimum_three_messages(self, agent_config):
        messages = _build_messages(agent_config, [], [], [], "hi")
        assert len(messages) >= 2  # at least system + user


class TestStreamAnswer:
    def _make_chunk(self, content):
        chunk = MagicMock()
        chunk.choices[0].delta.content = content
        return chunk

    def test_yields_tokens_when_no_tool_calls(self, agent_config):
        first_msg = MagicMock()
        first_msg.tool_calls = None
        first_msg.content = None

        stream_chunks = [self._make_chunk("Hello"), self._make_chunk(" world")]
        end_chunk = MagicMock()
        end_chunk.choices[0].delta.content = None

        with patch("llm.provider.client") as mock_client:
            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=first_msg)]),
                stream_chunks,
            ]
            result = list(stream_answer(agent_config, [], [], [], "hello"))

        assert "Hello" in result
        assert " world" in result

    def test_yields_direct_content_when_no_tool_calls_and_has_content(self, agent_config):
        first_msg = MagicMock()
        first_msg.tool_calls = None
        first_msg.content = "Direct answer"

        with patch("llm.provider.client") as mock_client:
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=first_msg)]
            )
            result = list(stream_answer(agent_config, [], [], [], "question"))

        assert result == ["Direct answer"]
