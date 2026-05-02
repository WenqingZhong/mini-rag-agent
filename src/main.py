"""
Entry Point — wiring all tiers together
=========================================
Production equivalent: ChatWebServiceImpl + StreamingChatService.handlePrompt()

WHAT'S NEW VS THE ORIGINAL:
  Tier 1 — Auth (X-API-Key), structured errors, persistent DB, /ingest endpoint
  Tier 2 — Tool calling, long-term memory, token-managed history
  Tier 3 — Prometheus /metrics, routing cache, embedding cache

FULL REQUEST PIPELINE:
  ① Auth           → verify X-API-Key header
  ② Select agent   → keyword → LLM (with routing cache)
  ③ Retrieve facts → embed question → ChromaDB similarity search (persistent)
  ④ Search memory  → find relevant long-term facts about this user
  ⑤ Load history   → last N messages from SQLite (trimmed to token budget)
  ⑥ LLM call       → tool detection → execute tools → stream final answer
  ⑦ Save history   → persist user + assistant messages to SQLite
  ⑧ Extract memory → async LLM pass to pull memorable facts from the turn

ENDPOINTS:
  POST /chat              → full JSON response
  POST /chat/stream       → SSE streaming
  POST /ingest            → add documents to agent knowledge base
  DELETE /conversation/{id} → clear history
  GET  /cache/stats       → cache hit rates
  GET  /health            → liveness check
  GET  /metrics           → Prometheus scrape (no auth)
"""

import uuid
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.selector import select_agent
from rag.retriever import retrieve, collection_size
from rag.ingestion import ingest_text
from llm.provider import stream_answer
from conversation.history import add_message, get_history, clear_history
from conversation.memory import search_memories, maybe_extract_and_store
from middleware.auth import verify_api_key
from middleware.errors import register_error_handlers
from middleware.metrics import (
    register_metrics,
    AGENT_SELECTIONS,
    FACTS_RETRIEVED,
    TOKEN_USAGE,
    LLM_ERRORS,
)
from utils.cache import embedding_cache, routing_cache
from utils.logger import get_logger
from config import API_KEY

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "\n\n❌ OPENAI_API_KEY is not set.\n" "Run: export OPENAI_API_KEY=sk-your-key\n"
    )

logger = get_logger("main")


# ── Startup: seed knowledge base ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_knowledge_base()
    yield


def _seed_knowledge_base():
    """
    Ingest demo documents if collections are empty.
    Because ChromaDB is now PERSISTENT, this only runs on first launch.
    On subsequent starts the data is already there.
    Production: IngestionRunService.startJob() triggers the loader microservice.
    """
    from rag.retriever import collection_size

    python_docs = [
        "Python list comprehensions: [x**2 for x in range(10)] creates [0,1,4,...,81]. "
        "They are faster and more Pythonic than equivalent for-loops.",
        "Virtual environments isolate project dependencies. Create: 'python -m venv venv'. "
        "Activate (Mac/Linux): 'source venv/bin/activate'. Activate (Windows): '.\\venv\\Scripts\\activate'.",
        "FastAPI uses Python type hints for automatic request/response validation via Pydantic. "
        "It auto-generates OpenAPI docs at /docs. Install: pip install fastapi uvicorn",
        "pytest is the standard Python testing framework. Test files must be named test_*.py. "
        "Test functions must start with test_. Run all tests: pytest. Fixtures: @pytest.fixture",
        "Python decorators wrap a function to add behaviour. Example: @staticmethod, @property. "
        "A decorator is a function that takes a function and returns a modified function.",
        "asyncio enables async/await concurrency. Use 'async def' for coroutines and 'await' "
        "to call them. FastAPI runs on an async event loop natively.",
    ]

    devops_docs = [
        "Docker builds portable container images. Build: 'docker build -t myapp .' "
        "Run: 'docker run -p 8000:8000 myapp'. List images: 'docker images'.",
        "A Dockerfile defines the build steps. Key instructions: FROM (base image), "
        "COPY (add files), RUN (execute commands during build), CMD (default run command).",
        "GitHub Actions automates CI/CD. Create .github/workflows/ci.yml. "
        "Trigger on push or pull_request. Each job runs steps sequentially on a runner.",
        "Kubernetes (k8s) orchestrates containers at scale. Key objects: Pod, Deployment "
        "(manages replicas), Service (exposes pods), Ingress (HTTP routing). Use kubectl.",
        "docker-compose defines multi-container stacks locally. 'docker compose up' starts all "
        "services. Define DB + app + cache in one docker-compose.yml file.",
    ]

    general_docs = [
        "RAG (Retrieval-Augmented Generation) combines vector search with LLM generation. "
        "Documents are chunked, embedded, and stored in a vector DB. At query time, "
        "similar chunks are retrieved and injected into the LLM prompt as context.",
        "REST APIs use HTTP methods: GET (read), POST (create), PUT (replace), "
        "PATCH (partial update), DELETE (remove). Responses are typically JSON.",
        "Git key commands: clone, add, commit, push, pull, branch, checkout, merge, rebase. "
        "Use 'git log --oneline' for a compact history view.",
        "A vector database stores high-dimensional embeddings and supports similarity search. "
        "Popular options: ChromaDB (local), Pinecone (cloud), Weaviate, Qdrant, pgvector.",
    ]

    seeded = False
    if collection_size("python_docs") == 0:
        ingest_text("python_docs", " ".join(python_docs), source="seed")
        seeded = True
    if collection_size("devops_docs") == 0:
        ingest_text("devops_docs", " ".join(devops_docs), source="seed")
        seeded = True
    if collection_size("general_docs") == 0:
        ingest_text("general_docs", " ".join(general_docs), source="seed")
        seeded = True

    if seeded:
        logger.info("Knowledge base seeded for first launch")
    else:
        logger.info("Knowledge base already populated — skipping seed")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mini RAG Agent",
    description="Educational RAG + agent-selection service. All tiers implemented.",
    version="2.0.0",
    lifespan=lifespan,
)

register_error_handlers(app)  # Tier 1: structured JSON error responses
register_metrics(app)  # Tier 3: Prometheus /metrics + latency middleware


# ── Request / Response models ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    user_id: str | None = "anonymous"  # for per-user memory


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    match_type: str
    conversation_id: str
    facts_retrieved: int


class IngestRequest(BaseModel):
    collection: str  # e.g. "python_docs"
    text: str  # raw document text
    source: str = "api"


class IngestResponse(BaseModel):
    collection: str
    chunks_added: int


# ── Helper: run the full pipeline ─────────────────────────────────────────────


def _run_pipeline(
    req: ChatRequest,
) -> tuple[str, dict, str, list[str], list[dict], list[str]]:
    """
    Shared setup for both /chat and /chat/stream.
    Returns: (conversation_id, agent, match_type, facts, history, memories)
    """
    conversation_id = req.conversation_id or str(uuid.uuid4())
    user_id = req.user_id or "anonymous"

    # ① Select agent (keyword → LLM, with cache)
    agent, match_type = select_agent(req.question)
    AGENT_SELECTIONS.labels(agent=agent["name"], match_type=match_type).inc()

    # ② Retrieve RAG facts
    facts = retrieve(agent["collection"], req.question)
    FACTS_RETRIEVED.labels(agent=agent["name"]).observe(len(facts))

    # ③ Search long-term memories
    memories = search_memories(user_id, req.question)

    # ④ Load conversation history (token budget trimming happens inside provider.py)
    history = get_history(conversation_id)

    return conversation_id, agent, match_type, facts, history, memories


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest):
    """
    Non-streaming: run full pipeline and return complete answer.
    Production equivalent: AskWebServiceImpl.ask()
    """
    conversation_id, agent, match_type, facts, history, memories = _run_pipeline(req)

    # Collect all streamed tokens into a single string
    answer = "".join(stream_answer(agent, facts, memories, history, req.question))

    # Persist this turn
    add_message(conversation_id, "user", req.question)
    add_message(conversation_id, "assistant", answer)

    # Extract memories in background thread (non-blocking)
    # Production: MemoryService.maybeStoreMemory() is called @Async
    full_conv = history + [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": answer},
    ]
    threading.Thread(
        target=maybe_extract_and_store,
        args=(req.user_id or "anonymous", full_conv),
        daemon=True,
    ).start()

    return ChatResponse(
        answer=answer,
        agent_used=agent["name"],
        match_type=match_type,
        conversation_id=conversation_id,
        facts_retrieved=len(facts),
    )


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_stream(req: ChatRequest):
    """
    Streaming: yield tokens as SSE as they arrive from the LLM.
    Production equivalent: ChatWebServiceImpl + Flux<ServerSentEvent<MessageRsp>>
    """
    conversation_id, agent, match_type, facts, history, memories = _run_pipeline(req)

    def generate():
        yield f"data: [AGENT:{agent['name']}|MATCH:{match_type}]\n\n"
        yield f"data: [FACTS:{len(facts)}|MEMORIES:{len(memories)}]\n\n"

        collected = []
        try:
            for token in stream_answer(agent, facts, memories, history, req.question):
                collected.append(token)
                yield f"data: {token}\n\n"
        except Exception as e:
            LLM_ERRORS.labels(error_type=type(e).__name__).inc()
            logger.error("LLM stream error", extra={"error": str(e)})
            yield f"data: [ERROR:{type(e).__name__}]\n\n"
            return

        full_answer = "".join(collected)
        add_message(conversation_id, "user", req.question)
        add_message(conversation_id, "assistant", full_answer)

        # Background memory extraction
        full_conv = history + [
            {"role": "user", "content": req.question},
            {"role": "assistant", "content": full_answer},
        ]
        threading.Thread(
            target=maybe_extract_and_store,
            args=(req.user_id or "anonymous", full_conv),
            daemon=True,
        ).start()

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post(
    "/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)]
)
def ingest_document(req: IngestRequest):
    """
    Ingest raw text into a named collection.
    Production equivalent: IngestionRunService.startJob() + DistributedLockService.acquireLock()

    Example:
      curl -X POST http://localhost:8000/ingest \\
        -H "X-API-Key: dev-secret-key-change-me" \\
        -H "Content-Type: application/json" \\
        -d '{"collection": "python_docs", "text": "Your document...", "source": "my-doc"}'
    """
    chunks = ingest_text(req.collection, req.text, source=req.source)
    logger.info(
        "Document ingested via API",
        extra={"collection": req.collection, "chunks": chunks},
    )
    return IngestResponse(collection=req.collection, chunks_added=chunks)


@app.delete("/conversation/{conversation_id}", dependencies=[Depends(verify_api_key)])
def delete_conversation(conversation_id: str):
    """Clear all history for a conversation."""
    deleted = clear_history(conversation_id)
    return {"conversation_id": conversation_id, "messages_deleted": deleted}


@app.get("/cache/stats", dependencies=[Depends(verify_api_key)])
def cache_stats():
    """Return hit/miss rates for all caches. Useful for tuning TTL values."""
    return {
        "embedding_cache": embedding_cache.stats(),
        "routing_cache": routing_cache.stats(),
    }


@app.get("/health")
def health():
    """Liveness check. No auth required (for load balancer probes)."""
    return {"status": "ok", "version": "2.0.0"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
