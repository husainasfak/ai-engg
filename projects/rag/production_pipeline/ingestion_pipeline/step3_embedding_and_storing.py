import hashlib
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from step2_chunking_and_metadata import Chunk

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")



def make_chunk_id(chunk: Chunk) -> str:
    """Create a repeatable ID from source metadata and chunk content."""
    source = str(chunk.metadata.get("source", "unknown"))
    page = str(chunk.metadata.get("page", ""))
    section = str(chunk.metadata.get("section", ""))
    index = str(chunk.metadata.get("chunk_index", ""))
    fingerprint = hashlib.sha1(chunk.content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{page}:{section}:{index}:{fingerprint}"

def build_index(
    chunks: list[Chunk],
    collection_name: str = "documents",
    embedding_model: str = "text-embedding-3-small",
) -> chromadb.Collection:
    """Embed chunks and store them in ChromaDB."""
    client = OpenAI(api_key=openai_api_key)
    chroma = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection for a clean rebuild. For live systems,
    # prefer incremental upserts with stable chunk IDs.
    try:
        chroma.delete_collection(collection_name)
    except ValueError:
        pass

    collection = chroma.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},  # Use cosine distance
    )

    # Embed in batches to avoid API limits
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.content for chunk in batch]

        # Get embeddings from OpenAI
        response = client.embeddings.create(
            input=texts,
            model=embedding_model,
        )
        embeddings = [item.embedding for item in response.data]

        # Prepare metadata (ChromaDB requires flat key-value pairs)
        metadatas = []
        for chunk in batch:
            meta = {}
            for key, value in chunk.metadata.items():
                # ChromaDB only supports str, int, float, bool
                meta[key] = str(value) if not isinstance(value, (int, float, bool)) else value
            metadatas.append(meta)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[make_chunk_id(chunk) for chunk in batch],
        )

    print(f"Indexed {len(chunks)} chunks into '{collection_name}'")
    return collection


