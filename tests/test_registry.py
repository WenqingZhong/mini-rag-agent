"""
Unit tests for agents/registry.py

Tests agent lookup, listing, and error handling. Pure dict operations — no external calls.
"""

import pytest
from agents.registry import get_agent, list_agents, AGENT_REGISTRY


class TestGetAgent:
    def test_returns_python_agent(self):
        agent = get_agent("python-agent")
        assert agent["name"] == "python-agent"
        assert agent["collection"] == "python_docs"

    def test_returns_devops_agent(self):
        agent = get_agent("devops-agent")
        assert agent["name"] == "devops-agent"
        assert agent["collection"] == "devops_docs"

    def test_returns_general_agent(self):
        agent = get_agent("general-agent")
        assert agent["name"] == "general-agent"

    def test_unknown_agent_raises_key_error(self):
        with pytest.raises(KeyError, match="not found"):
            get_agent("nonexistent-agent")

    def test_returned_agent_has_required_fields(self):
        agent = get_agent("python-agent")
        for field in ("name", "description", "system_prompt", "collection", "keywords"):
            assert field in agent, f"Missing field: {field}"

    def test_returned_agent_is_from_registry(self):
        # get_agent returns the same dict object as the registry holds
        assert get_agent("python-agent") is AGENT_REGISTRY["python-agent"]


class TestListAgents:
    def test_returns_list(self):
        assert isinstance(list_agents(), list)

    def test_returns_all_agents(self):
        agents = list_agents()
        names = {a["name"] for a in agents}
        assert "python-agent" in names
        assert "devops-agent" in names
        assert "general-agent" in names

    def test_count_matches_registry(self):
        assert len(list_agents()) == len(AGENT_REGISTRY)

    def test_each_agent_has_keywords_list(self):
        for agent in list_agents():
            assert isinstance(agent.get("keywords"), list)

    def test_general_agent_has_no_keywords(self):
        general = next(a for a in list_agents() if a["name"] == "general-agent")
        assert general["keywords"] == []
