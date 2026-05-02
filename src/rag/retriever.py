"""
Vector store repository — the read/write layer for ChromaDB.

This is the lowest-level RAG component. It owns the ChromaDB connection
and exposes two operations that mirror each other:
  ingest()   → embed chunks and write them to a collection (called at startup/ingest)
  retrieve() → embed a question and find the closest chunks (called on every request)

Each agent gets its own collection (namespace) so their knowledge bases don't mix.
"""

import uuid
import os
import chromadb
from rag.embedder import embed
from utils.logger import get_logger
from config import CHROMA_PATH, DATA_DIR, TOP_K_FACTS

logger = get_logger("retriever")

# Single persistent ChromaDB client shared across all collections.
# PersistentClient writes to disk so data survives restarts.
os.makedirs(DATA_DIR, exist_ok=True)
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)

# In-process cache of collection handles so we don't re-fetch from ChromaDB each call.
_collections: dict[str, chromadb.Collection] = {}


def _get_collection(name: str) -> chromadb.Collection:
    """
    Return the named ChromaDB collection, creating it if it doesn't exist yet.

    cosine similarity is used instead of the default L2 distance because it
    measures the angle between vectors (topic similarity) rather than their
    magnitude — better suited for text embeddings.
    """
    if name not in _collections:
        _collections[name] = _chroma.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[name]


def ingest(
    collection_name: str,
    texts: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """
    Embed each text chunk and write it to ChromaDB.

    Each chunk gets a random UUID as its ID. Metadatas (source, chunk_index)
    are stored alongside the vector so we can trace results back to their origin.

    Note: ChromaDB 0.5+ rejects empty metadata dicts, so we only pass
    metadatas when the caller explicitly provides them.
    """
    if not texts:
        return

    collection = _get_collection(collection_name)
    ids = [str(uuid.uuid4()) for _ in texts]
    embeddings = [embed(t) for t in texts]

    add_kwargs: dict = dict(documents=texts, embeddings=embeddings, ids=ids)
    if metadatas:
        add_kwargs["metadatas"] = metadatas

    collection.add(**add_kwargs)
    logger.info(
        "Ingested chunks", extra={"collection": collection_name, "count": len(texts)}
    )


def retrieve(
    collection_name: str,
    question: str,
    top_k: int = TOP_K_FACTS,
) -> list[str]:
    """
    Find the top-K chunks most semantically similar to the question.

    Flow:
      1. Embed the question into a vector
      2. Ask ChromaDB to find the closest stored vectors (cosine similarity)
      3. Return the original text of those chunks as "Facts" for the LLM prompt

    top_k is capped at the actual collection size to avoid ChromaDB errors
    when fewer chunks exist than requested.
    """
    collection = _get_collection(collection_name)

    if collection.count() == 0:
        logger.warning(
            "Collection empty — no facts retrieved",
            extra={"collection": collection_name},
        )
        return []

    query_vec = embed(question)
    n = min(top_k, collection.count())  # can't request more results than exist
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "distances"],
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]

    logger.info(
        "Facts retrieved",
        extra={
            "collection": collection_name,
            "count": len(chunks),
            "distances": [round(d, 4) for d in distances],
        },
    )
    return chunks


def collection_size(collection_name: str) -> int:
    """Return the total number of stored chunks in a collection."""
    return _get_collection(collection_name).count()
