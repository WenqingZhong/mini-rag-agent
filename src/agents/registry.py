"""
Agent registry — the single source of truth for all agent definitions.

An agent is a named configuration that controls three things:
  1. system_prompt — the persona and instructions injected into every LLM call
  2. collection    — which ChromaDB namespace holds this agent's knowledge
  3. keywords      — trigger words for fast routing before the LLM is consulted

Adding a new agent means adding one entry to AGENT_REGISTRY.
No other files need to change.
"""

# Dict keyed by agent name for O(1) lookups by name.
AGENT_REGISTRY: dict[str, dict] = {
    "python-agent": {
        "name": "python-agent",
        "description": "Answers questions about Python programming, syntax, libraries, and best practices.",
        "system_prompt": (
            "You are an expert Python programmer."
            "Use the provided context to answer questions accurately."
            "Always include working code examples when relevant."
            "If the context does not cover the question, say so clearly."
        ),
        "collection": "python_docs",
        "keywords": [
            "python",
            "pip",
            "venv",
            "fastapi",
            "pandas",
            "numpy",
            "pytest",
            "decorator",
            "asyncio",
            "programming",
            "code",
            "syntax",
            "libraries",
        ],
    },
    "devops-agent": {
        "name": "devops-agent",
        "description": "Provides guidance on DevOps tools, CI/CD pipelines, and infrastructure management.",
        "system_prompt": (
            "You are a DevOps expert with deep knowledge of CI/CD pipelines, containerization, and cloud infrastructure."
            "Use the provided context to answer questions accurately."
            "Always include practical examples when relevant."
            "If the context does not cover the question, say so clearly."
        ),
        "collection": "devops_docs",
        "keywords": [
            "devops",
            "ci/cd",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "terraform",
            "infrastructure",
            "automation",
            "monitoring",
        ],
    },
    "general-agent": {
        "name": "general-agent",
        "description": "Fallback agent for questions not covered by any specialist agent.",
        "system_prompt": (
            "You are a helpful assistant. "
            "Use the provided context to answer questions accurately. "
            "If the context does not contain enough information, answer from general knowledge "
            "but clearly state that the answer is not based on the provided context."
        ),
        "collection": "general_docs",
        "keywords": [],  # no keywords — this agent is only reached via LLM routing or fallback
    },
}


def get_agent(agent_name: str) -> dict:
    """
    Look up an agent by name and return its full config dict.

    Raises KeyError (caught by the global error handler → HTTP 404) if the
    name doesn't exist in the registry.
    """
    if agent_name not in AGENT_REGISTRY:
        raise KeyError(
            f"Agent '{agent_name}' not found. Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[agent_name]


def list_agents() -> list[dict]:
    """Return all agent configs as a list (used by the selector to build the routing prompt)."""
    return list(AGENT_REGISTRY.values())
