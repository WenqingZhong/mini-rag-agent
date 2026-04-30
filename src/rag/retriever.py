"""
CONCEPT: RAG Retriever

This is the heart of the RAG pipeline. It does two things:
1. INGEST (offline, setup time):
    raw text -> chunking -> embedding -> store in ChromeDB with metadata (agent collection, source doc, etc)
2. RETRIEVE (online, query time):
    user question -> embedding -> vector search in ChromeDB -> return top K chunks

"""

import uuid
import chromadb
from rag.embedder import embed

# in memory chromadb client - in production this would be a remote service
_chromadb = chromadb.Client()

# Cache collection handles so we don't re-fetch them every time
_collections: dict[str, chromadb.Collection] = {}

def _get_collection(name: str):
    if name not in _collections:
        _collections[name] = _chromadb.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
         ) # use cosine for better recall at the cost of precision - good for RAG retriever)
    return _collections[name]

def ingest(collection_name: str, text_chunks: list[str], metadata_list: list[dict] | None = None):
    """
    Ingest a list of text chunks into the specified collection, along with optional metadata.
    Each chunk will be embedded and stored as a vector in ChromeDB.
    Call this once at startup (or as background job in production)
    Args:
        collection_name (str): The name of the collection to store the chunks in (should correspond to agent collection).
        text_chunks (list[str]): The list of text chunks to be ingested.
        metadata_list (list[dict], optional): A list of metadata dicts corresponding to each chunk. Defaults to None.
    """
    if not text_chunks:
        return
    
    collection = _get_collection(collection_name)

    print(f"[Retriever] Ingesting {len(text_chunks)} chunks into collection '{collection_name}'")

    embeddings = [embad(chunk) for chunk in text_chunks]
    ids = [str(uuid.uuid4()) for _ in text_chunks]

    # documents = raw text chunks
    # embeddings = vector representations of the chunks
    # metadatas = any additional info we want to store (e.g. source doc). Only pass in if the caller explicitly provides non-empty ones
    # Chorme DB uses vectors to find matches and return the original text as facts
    add_kwargs: dict = dict(
        ids=ids,
        embeddings=embeddings,
        documents=text_chunks,
    )
    if metadata_list:
        add_kwargs["metadatas"] = metadata_list
    
    collection.add(**add_kwargs)
    print(f"[Retriever] Successfully ingested {len(text_chunks)} chunks into collection '{collection_name}'")



    def retrieve(collection_name: str, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve the top K most relevant chunks from the specified collection based on the query.
        Call this on every user question to get the relevant context for the LLM.
        Args:
            collection_name (str): The name of the collection to search in (should correspond to agent collection).
            query (str): The user question or query string to find relevant chunks for.
            top_k (int, optional): The number of top relevant chunks to return. Defaults to 5.
        Returns:
            list[str]: A list of the top K most relevant text chunks retrieved from the collection.
        """
        collection = _get_collection(collection_name)
        if collection.count() == 0:
            print(f"[Retriever] Warning: Collection '{collection_name}' is empty. No chunks to retrieve.")
            return []
        
        query_embedding = embed(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],  # we only need the original text chunks as output, not the embeddings or metadata
        )

        retrieved_chunks = results["documents"][0]  # list of top K chunks
        retrieved_distances = results["distances"][0]  # list of cosine distances for the top K chunks (lower = more similar)

        print(f"[Retriever] Retrieved {len(retrieved_chunks)} chunks from collection '{collection_name}' for query: {query[:80]}...")
        for i, (chunk, distance) in enumerate(zip(retrieved_chunks, retrieved_distances)):
            print(f"Fact {i+1}. (distance: {distance:.4f}) {chunk[:80]}...")

        return retrieved_chunks