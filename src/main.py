"""
CONCEPT: Entry Point — HTTP Layer + Pipeline Orchestration
===========================================================
Production equivalent: ChatWebServiceImpl (HTTP routing)
                       + StreamingChatService.handlePrompt() (pipeline)

This file wires ALL the pieces together. Every request flows through
the same pipeline — this is the "main loop" of a RAG agent service:

  ┌─────────────────────────────────────────────────────────┐
  │                  RAG AGENT PIPELINE                     │
  │                                                         │
  │  User Question                                          │
  │       │                                                 │
  │       ▼                                                 │
  │  [1] selector.py   → Which agent handles this?          │
  │       │                                                 │
  │       ▼                                                 │
  │  [2] retriever.py  → Fetch relevant facts from ChromaDB │
  │       │                                                 │
  │       ▼                                                 │
  │  [3] history.py    → Load prior conversation turns      │
  │       │                                                 │
  │       ▼                                                 │
  │  [4] provider.py   → Build prompt + stream LLM response │
  │       │                                                 │
  │       ▼                                                 │
  │  [5] history.py    → Save user msg + assistant response │
  │       │                                                 │
  │       ▼                                                 │
  │  SSE Stream → Client                                    │
  └─────────────────────────────────────────────────────────┘

TWO ENDPOINTS:
  POST /chat         → Returns full response (JSON). Good for testing.
  POST /chat/stream  → Returns Server-Sent Events. Good for UI integration.

DEMO KNOWLEDGE BASE:
  On startup we ingest a few hard-coded text chunks into ChromaDB.
  In production this is done by IngestionRunService reading from
  real sources: Confluence, GitHub, Jira, GCS files, etc.
"""

import uuid
import os
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.selector import select_agent
from rag.retriever import ingest, retriever
from llm.provider import stream_answer
from conversation.history import add_message, get_history, clear_history

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "\n\n❌ OPENAI_API_KEY is not set.\n"
        "Fix: run this in your terminal before starting the server:\n\n"
        "  export OPENAI_API_KEY=sk-your-key-here\n\n"
        "Then run: python main.py\n"
    )

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=sk-...")


# ---------------------------------------------------------------------------
# Startup: seed the knowledge base
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup to ingest demo documents into ChromaDB.
    Production: IngestionRunService handles this via scheduled jobs.
    """
    _seed_knowledge_base()
    yield  # server runs here
    # (cleanup code would go after yield if needed)


def _seed_knowledge_base():
    """
    Ingest demo documents so the agents have something to retrieve.

    In production: these come from real data sources — Confluence pages,
    GitHub READMEs, Jira tickets, GCS files, internal wikis, etc.
    Here: plain strings. The RAG mechanics are identical.
    """
    print("\n[Startup] Seeding knowledge base...")

    # ── Python Agent knowledge ─────────────────────────────────────────────
    python_docs = [
        "Python list comprehensions provide a concise way to create lists. "
        "Example: squares = [x**2 for x in range(10)] produces [0, 1, 4, 9, ..., 81].",

        "Virtual environments isolate project dependencies. Create one with: "
        "'python -m venv venv', then activate it: 'source venv/bin/activate' (Mac/Linux) "
        "or '.\\venv\\Scripts\\activate' (Windows). Install packages inside the venv.",

        "FastAPI is a modern Python web framework for building APIs. It uses type hints "
        "for automatic validation and generates OpenAPI docs at /docs. "
        "Install with: pip install fastapi uvicorn",

        "pytest is the standard Python testing framework. Name test files test_*.py "
        "and test functions test_*(). Run all tests with: pytest. "
        "Use fixtures for shared setup: @pytest.fixture",

        "Python decorators wrap functions to add behavior without modifying them. "
        "Example: @staticmethod, @property, @app.route('/path'). "
        "A decorator is just a function that takes a function and returns a function.",

        "asyncio enables async/await concurrency in Python. Use 'async def' to define "
        "coroutines and 'await' to call them. FastAPI, aiohttp, and SQLAlchemy support async.",
    ]

    # ── DevOps Agent knowledge ─────────────────────────────────────────────
    devops_docs = [
        "Docker containers package code and all dependencies into a portable unit. "
        "Build an image: 'docker build -t myapp .' Run it: 'docker run -p 8000:8000 myapp'",

        "A Dockerfile defines how to build a container image. Key instructions: "
        "FROM (base image), COPY (add files), RUN (execute commands), CMD (default command). "
        "Example: FROM python:3.11-slim",

        "GitHub Actions automates CI/CD workflows. Create a file at .github/workflows/ci.yml. "
        "Triggers: on: [push, pull_request]. Steps run in sequence inside a job.",

        "Kubernetes (k8s) orchestrates containers at scale. Key objects: "
        "Pod (smallest unit), Deployment (manages replicas), Service (exposes pods), "
        "Ingress (HTTP routing). Use kubectl to interact with the cluster.",

        "docker-compose.yml defines multi-container apps locally. "
        "Run with: 'docker compose up'. Define services, networks, and volumes. "
        "Great for local development with a DB + app + cache stack.",
    ]

    # ── General Agent knowledge ────────────────────────────────────────────
    general_docs = [
        "RAG (Retrieval-Augmented Generation) combines a vector search step with LLM generation. "
        "Documents are chunked, embedded, and stored in a vector DB. At query time, "
        "the question is embedded and similar chunks are retrieved to give the LLM context.",

        "REST APIs use HTTP methods: GET (read), POST (create), PUT (replace), "
        "PATCH (partial update), DELETE (remove). Responses are typically JSON.",

        "Git is a distributed version control system. Key commands: "
        "git clone, git add, git commit, git push, git pull, git branch, git merge.",

        "A vector database stores high-dimensional embeddings and supports similarity search. "
        "Popular options: ChromaDB (local), Pinecone (cloud), Weaviate, Qdrant.",
    ]

    ingest("python_docs", python_docs)
    ingest("devops_docs",  devops_docs)
    ingest("general_docs", general_docs)

    print("[Startup] ✓ Knowledge base ready.\n")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mini RAG Agent",
    description="A minimal RAG + agent-selection service for learning.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None  # None = start a new conversation


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    conversation_id: str
    facts_retrieved: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Non-streaming chat endpoint.
    Runs the full RAG pipeline and returns the complete answer as JSON.

    Good for: testing, curl, Postman, simple integrations.
    Production equivalent: AskWebServiceImpl.ask()
    """
    conversation_id = req.conversation_id or str(uuid.uuid4())

    # ── Step 1: Select agent ─────────────────────────────────────────────
    agent = select_agent(req.question)

    # ── Step 2: Retrieve relevant facts (RAG) ───────────────────────────
    facts = retriever(agent["collection"], req.question)

    # ── Step 3: Load conversation history ───────────────────────────────
    history = get_history(conversation_id)

    # ── Step 4: Generate answer (collect all stream chunks) ──────────────
    answer_chunks = list(stream_answer(agent, facts, history, req.question))
    full_answer   = "".join(answer_chunks)

    # ── Step 5: Save to history ──────────────────────────────────────────
    add_message(conversation_id, "user",      req.question)
    add_message(conversation_id, "assistant", full_answer)

    return ChatResponse(
        answer          = full_answer,
        agent_used      = agent["name"],
        conversation_id = conversation_id,
        facts_retrieved = len(facts),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Yields tokens as they arrive from the LLM.

    Good for: UI integrations where you want to show typing-effect responses.
    Production equivalent: ChatWebServiceImpl streaming with Flux<ServerSentEvent>

    SSE format:  data: <token>\\n\\n
    Special events:
      data: [AGENT:agent-name]     ← which agent was selected
      data: [FACTS:N retrieved]    ← how many RAG chunks were found
      data: [DONE]                 ← stream complete
    """
    conversation_id = req.conversation_id or str(uuid.uuid4())

    def generate():
        # ── Step 1: Select agent ─────────────────────────────────────────
        agent = select_agent(req.question)
        yield f"data: [AGENT:{agent['name']}]\n\n"

        # ── Step 2: Retrieve facts ───────────────────────────────────────
        facts = retriever(agent["collection"], req.question)
        yield f"data: [FACTS:{len(facts)} retrieved]\n\n"

        # ── Step 3: Load history ─────────────────────────────────────────
        history = get_history(conversation_id)

        # ── Step 4: Stream LLM response token by token ───────────────────
        collected = []
        for token in stream_answer(agent, facts, history, req.question):
            collected.append(token)
            yield f"data: {token}\n\n"

        # ── Step 5: Save to history after full response arrives ──────────
        add_message(conversation_id, "user",      req.question)
        add_message(conversation_id, "assistant", "".join(collected))

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    """
    Clear conversation history. Lets the user "start fresh."
    Production equivalent: ConversationManagementService.deleteConversation()
    """
    clear_history(conversation_id)
    return {"message": f"Conversation '{conversation_id}' cleared."}


@app.get("/health")
def health():
    """Simple health check. Production: Spring Actuator /actuator/health"""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)