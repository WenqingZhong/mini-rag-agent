"""
Embedding — converts text into a vector (list of floats).

Vectors are what make semantic search possible: instead of matching keywords,
ChromaDB compares vectors by cosine similarity to find conceptually close chunks.
Results are cached so repeated text (e.g. the same query) never hits the API twice.
"""

from openai import OpenAI
from utils.cache import embedding_cache
from utils.logger import get_logger
from config import EMBEDDING_MODEL

logger = get_logger("embedder")
client = OpenAI()


def embed(text: str) -> list[float]:
    """
    Convert a string into an embedding vector.

    Flow:
      1. Normalise the text (collapse newlines, strip whitespace)
      2. Check the in-memory cache — return immediately on hit
      3. On miss: call the OpenAI Embeddings API and cache the result
    """
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        raise ValueError("Cannot embed empty text.")

    # Return cached vector if available (avoids redundant API calls)
    cached = embedding_cache.get(clean_text)
    if cached is not None:
        logger.debug("Embedding cache HIT", extra={"text_preview": clean_text[:50]})
        return cached

    logger.debug(
        "Embedding cache MISS - calling API", extra={"text_preview": clean_text[:50]}
    )
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=clean_text,
    )
    vector = response.data[0].embedding
    embedding_cache.set(clean_text, value=vector)
    return vector
