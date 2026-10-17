def incremental_update(
    collection: chromadb.Collection,
    updated_file: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
):
    """Re-index a single file without rebuilding the entire collection."""
    # Delete old chunks from this file
    existing = collection.get(where={"source": updated_file})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info(f"Deleted {len(existing['ids'])} old chunks from {updated_file}")

    # Load, chunk, and embed the updated file
    docs = load_document(updated_file)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))

    if chunks:
        # Re-index
        client = OpenAI()
        texts = [c.content for c in chunks]
        embeddings = embed_with_retry(texts)

        new_ids = [make_chunk_id(chunk) for chunk in chunks]
        metadatas = [{k: str(v) if not isinstance(v, (int, float, bool)) else v
                      for k, v in c.metadata.items()} for c in chunks]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=new_ids,
        )
        logger.info(f"Added {len(chunks)} new chunks from {updated_file}")