
import os
from typing import Optional

import chromadb

from dotenv import load_dotenv
from openai import OpenAI
    
load_dotenv()

openai_api_key =    os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

def retrieve(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    embedding_model: str = "text-embedding-3-small",
    # Metadata filtering. The where parameter lets you narrow the search scope before vector similarity is applied. For example, where={"format": "pdf"} only searches PDF-sourced chunks. This is useful when users specify a document type or when your application already knows which source is most relevant.
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query: The search query
        collection: ChromaDB collection to search
        n_results: Number of chunks to retrieve
        embedding_model: Must match the model used during ingestion
        where: Optional metadata filter (e.g., {"format": "pdf"})

    Returns:
        List of dicts with 'content', 'metadata', and 'score' keys
    """
    client = OpenAI(api_key=openai_api_key)

    # Embed the query using the same model as ingestion
    q_embedding = client.embeddings.create(
        input=query,
        model=embedding_model,
    ).data[0].embedding

    # Query ChromaDB
    query_params = {
        "query_embeddings": [q_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_params["where"] = where

    results = collection.query(**query_params)

    # Package results into a clean format
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            # For cosine distance in Chroma, smaller is better. This rough
            # conversion is useful for display, not calibrated confidence.
            "score": 1 - results["distances"][0][i],
        })

    return retrieved