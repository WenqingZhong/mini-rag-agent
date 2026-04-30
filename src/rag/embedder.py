"""
CONCEPT: Embedding

Turn text into vectors so that we can do vector search in ChromeDB for RAG.
"""
import os
from openai import OpenAI

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small" # cheap embedding model - rounting doesn't need high quality embeddings

def embed(text: str) -> list[float]:
    """
    Main entry point. Takes a string and returns its embedding vector.
    """
    clean_text = text.replace("\n", " ").strip()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=clean_text
    )
    
    return response.data[0].embedding