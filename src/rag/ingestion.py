"""
Ingestion pipeline — takes raw text or files and stores them in ChromaDB.

This is the write side of RAG. It sits above retriever.py:
  raw text → chunk_text() → ingest() in retriever.py → ChromaDB
"""

import os
from rag.retriever import ingest
from utils.logger import get_logger
from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

logger = get_logger("ingestion")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split a long text into overlapping word-level chunks.

    Why overlap? The answer to a question might straddle a chunk boundary.
    Overlap ensures that context around the boundary appears in at least one chunk.

    Example with chunk_size=5, overlap=2:
      "a b c d e f g h" → ["a b c d e", "d e f g h"]
                                  ↑↑ overlap keeps 'd e' in both chunks

    Returns a single-element list if the text fits in one chunk.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_size - overlap)  # how far to advance the window each iteration
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += step

    logger.info(
        "Text chunked",
        extra={
            "total_words": len(words),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    )
    return chunks


def ingest_text(
    collection_name: str,
    text: str,
    source: str = "manual",
) -> int:
    """
    Chunk a raw string and store all chunks in ChromaDB.

    This is the main entry point for adding knowledge to an agent's collection.
    Called from the /ingest endpoint and from _seed_knowledge_base() on startup.

    Returns the number of chunks stored.
    """
    chunks = chunk_text(text)
    # Tag each chunk with its source and position for traceability
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    ingest(collection_name, chunks, metadatas)
    logger.info(
        "Ingestion complete",
        extra={"collection": collection_name, "chunks": len(chunks), "source": source},
    )
    return len(chunks)


def ingest_file(collection_name: str, file_path: str) -> int:
    """
    Read a text file from disk and ingest its contents.

    Convenience wrapper around ingest_text() — reads the file then delegates.
    The file path is stored as the source tag on each chunk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return ingest_text(collection_name, text, source=file_path)
