"""
Shared test configuration and fixtures.

Module-level env vars are set HERE, before any src/ imports, so that
config.py reads temp paths instead of the real data directory.
"""

import os
import sys
import tempfile

import pytest

# ── Temp data directory for all test runs ────────────────────────────────────
# Must be set before any src/ module is imported, because config.py reads
# these at import time.
_test_dir = tempfile.mkdtemp(prefix="mini_rag_test_")

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-not-real")
os.environ["DATA_DIR"] = _test_dir
os.environ["CHROMA_PATH"] = os.path.join(_test_dir, "chroma")
os.environ["SQLITE_PATH"] = os.path.join(_test_dir, "conversations.db")
os.environ["MEMORY_DB_PATH"] = os.path.join(_test_dir, "memory.db")
os.environ["API_KEY"] = "test-api-key"

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def agent_config():
    """A minimal valid agent config dict."""
    return {
        "name": "python-agent",
        "description": "Python expert agent.",
        "system_prompt": "You are a Python expert.",
        "collection": "python_docs",
        "keywords": ["python", "pip"],
    }


@pytest.fixture
def sample_history():
    """Two turns of conversation history in OpenAI message format."""
    return [
        {"role": "user", "content": "What is a decorator?"},
        {"role": "assistant", "content": "A decorator wraps a function."},
    ]
