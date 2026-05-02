"""
Unit tests for agents/selector.py

Tests keyword matching (no external calls) and LLM matching (OpenAI mocked).
The routing cache is cleared between tests to prevent cross-test state.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.selector import select_agent, _keyword_match, _llm_match, FALLBACK_AGENT
from utils.cache import routing_cache


@pytest.fixture(autouse=True)
def clear_routing_cache():
    """Ensure the routing cache is empty before each test."""
    routing_cache.clear()
    yield
    routing_cache.clear()


class TestKeywordMatch:
    def test_python_keyword_matches_python_agent(self):
        assert _keyword_match("how do I use python?") == "python-agent"

    def test_docker_keyword_matches_devops_agent(self):
        assert _keyword_match("how do I build a docker image?") == "devops-agent"

    def test_kubernetes_keyword_matches_devops_agent(self):
        assert _keyword_match("kubernetes deployment failed") == "devops-agent"

    def test_no_keyword_match_returns_none(self):
        assert _keyword_match("what is the meaning of life?") is None

    def test_case_insensitive_match(self):
        assert _keyword_match("How do I use PYTHON?") == "python-agent"

    def test_pip_keyword_matches_python_agent(self):
        assert _keyword_match("pip install fails") == "python-agent"


class TestSelectAgentKeywordPath:
    def test_keyword_question_returns_correct_agent(self):
        agent, match_type = select_agent("how do I use python decorators?")
        assert agent["name"] == "python-agent"
        assert match_type == "keyword"

    def test_match_type_is_keyword(self):
        _, match_type = select_agent("docker build command")
        assert match_type == "keyword"

    def test_result_is_cached_after_first_call(self):
        select_agent("how to use python list comprehensions?")
        stats = routing_cache.stats()
        assert stats["size"] == 1


class TestSelectAgentCachePath:
    def test_cached_result_returned_on_second_call(self):
        question = "how do I write python code?"
        agent1, _ = select_agent(question)
        agent2, match_type = select_agent(question)
        assert match_type == "cached"
        assert agent1["name"] == agent2["name"]

    def test_cache_hit_increments_cache_stats(self):
        question = "python venv setup"
        select_agent(question)  # populates cache
        select_agent(question)  # cache hit
        assert routing_cache.stats()["hits"] >= 1


class TestSelectAgentLLMPath:
    def test_llm_match_called_when_no_keyword(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "general-agent"

        with patch("agents.selector.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            agent, match_type = select_agent("what is the airspeed of an unladen swallow?")

        assert match_type == "llm"
        assert agent["name"] == "general-agent"

    def test_invalid_llm_response_falls_back_to_general(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not-a-real-agent"

        with patch("agents.selector.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            agent, match_type = select_agent("something with no keyword match here")

        assert agent["name"] == FALLBACK_AGENT

    def test_llm_exception_falls_back_to_general(self):
        with patch("agents.selector.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("API error")
            agent, match_type = select_agent("question with no keyword match either")

        assert agent["name"] == FALLBACK_AGENT
