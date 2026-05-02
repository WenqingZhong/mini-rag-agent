"""
CONCEPT: Agent Registry
DESCRIPTION:
In production, agents are stored in a database with hundreds of config fields. 
Here we use a plain dict - the core idea is exactly the same: 
 An agent is just a named config that drives the rest of the pipeline.

Each agent has:
- name: unique identifier for the agent
- description: a brief description of the agent's purpose and capabilities
- system_prompt: the agent's personality and behavior guidelines that is injected into every LLM call
- collection : the ChromeDB collection where THIS agent's knowledge lives. (its RAG namespace)
- keywords: trigger words for fast routing BEFORE we call the LLM
"""

AGENT_REGISTRY : dict[str, dict] = {
    "python-agent": {
        "name": "python-agent",
        "description": "An agent that answers questions about Python programming, syntax, libraries, and best practices.",
        "system_prompt": (
            "You are an expert Python programmer."
            "Use the provided context to answer questions accurately."
            "Always include working code examples when relevant."
            "If the context does not cover the question, say so clearly."
        ),
        "collection": "python_docs",
        "keywords": ["python", "pip", "venv", "fastapi", "pandas", "numpy", "pytest", "decorator", "asyncio", 
                     "programming", "code", "syntax", "libraries"],
    },
    "devops-agent": {
        "name": "devops-agent",
        "description": "An agent that provides information and best practices on DevOps tools, CI/CD pipelines, and infrastructure management.",
        "system_prompt": (
            "You are a DevOps expert with deep knowledge of CI/CD pipelines, containerization, and cloud infrastructure."
            "Use the provided context to answer questions accurately."
            "Always include practical examples when relevant."
            "If the context does not cover the question, say so clearly."
        ),
        "collection": "devops_docs",
        "keywords": ["devops", "ci/cd", "docker", "kubernetes", "aws", "azure", "terraform", 
                     "infrastructure", "automation", "monitoring"],
    },
    "general-agent": {
        "name": "general-agent",
        "description": "A general-purpose agent that answer questions that are not covered by the more specialized agents.",
        "system_prompt": (
            "You are a helpful assistant. "
            "Use the provided context to answer questions accurately."
            "If the context does not contain enough information to answer the question, answer to the best of your ability based on your general knowledge, "
            "but clearly indicate that the answer is not based on the provided context."
        ),
        "collection": "general_docs",
        "keywords": [],
    },
}


def get_agent(agent_name: str) -> dict:
    """
    Retrieve the configuration for a specific agent by name.
    Args:
        agent_name (str): The unique identifier for the agent.
    Returns:
        dict: The configuration for the specified agent, or an error message if the agent is not found.
    """
    if agent_name not in AGENT_REGISTRY:
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_name]

def list_agents() -> list[dict]:
    """
    List all available agents in the registry.
    Returns:
        list[dict]: A list of configurations for all registered agents.
    """
    return list(AGENT_REGISTRY.values())
