def assemble_context(retrieved_chunks: list[dict], max_tokens: int = 3000) -> str:
    """
    Format retrieved chunks into a context string for the model.

    Includes source information for citations.
    Respects a rough token budget to avoid overflowing the context window.
    """
    context_parts = []
    estimated_tokens = 0

    for i, chunk in enumerate(retrieved_chunks):
        # Build source citation
        meta = chunk["metadata"]
        source_info = f"Source: {meta.get('source', 'unknown')}"
        if "page" in meta:
            source_info += f", Page {meta['page']}"
        if "section" in meta:
            source_info += f", Section: {meta['section']}"

        # Format the chunk with its citation
        formatted = f"[{i + 1}] {source_info}\n{chunk['content']}"

        # Rough token estimate (1 token ~ 4 characters)
        chunk_tokens = len(formatted) // 4
        if estimated_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(formatted)
        estimated_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)