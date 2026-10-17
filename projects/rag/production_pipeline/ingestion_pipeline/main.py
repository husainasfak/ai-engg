import os
from pathlib import Path

import chromadb

from step1_document_loading import load_document
from step2_chunking_and_metadata import chunk_document
from step3_embedding_and_storing import build_index

def ingest(
    source_dir: str,
    collection_name: str = "documents",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> chromadb.Collection:
    """
    Full ingestion pipeline: load, chunk, embed, and store documents.

    Args:
        source_dir: Directory containing documents to ingest
        collection_name: Name for the ChromaDB collection
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        ChromaDB collection with indexed chunks
    """
    # Step 1: Load all documents
    all_documents = []
    supported_extensions = {".pdf", ".md", ".html", ".htm", ".txt"}

    for file_path in Path(source_dir).rglob("*"):
        if file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_document(str(file_path))
                all_documents.extend(docs)
                print(f"Loaded: {file_path.name} ({len(docs)} sections)")
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")

    # Step 2: Chunk all documents
    all_chunks = []
    for doc in all_documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    # Step 3: Embed and store
    collection = build_index(all_chunks, collection_name)

    return collection


# Ingest all documents from a directory
collection = ingest(
    source_dir="./docs",
    collection_name="product_docs",
    chunk_size=800,
    chunk_overlap=200,
)