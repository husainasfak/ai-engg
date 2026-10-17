"""
Chunking creates a tradeoff. Small chunks are easier to match precisely, but they may not include enough context. Large chunks preserve context, but they can be harder to retrieve accurately.

Parent-child retrieval solves a chunking problem in RAG.
Search using a small “child” chunk, but give the larger “parent” section to the LLM.

Small chunks → better retrieval, less context
Large chunks → more context, less precise retrieval
Parent-child → best of both


"""

def index_parent_child(parents: list[dict], vector_store):
    """
    Store small child chunks for retrieval and keep parent text by parent_id.
    Each parent has: id, content, metadata, children.
    """
    child_records = []

    for parent in parents:
        for i, child_text in enumerate(parent["children"]):
            child_records.append({
                "id": f"{parent['id']}::child::{i}",
                "content": child_text,
                "metadata": {
                    **parent["metadata"],
                    "parent_id": parent["id"],
                    "child_index": i,
                },
            })

    vector_store.add(child_records)

def retrieve_parents(question: str, vector_store, parent_store, top_k: int = 5):
    child_hits = vector_store.search(question, top_k=top_k * 3)

    parent_ids = []
    seen = set()
    for hit in child_hits:
        parent_id = hit["metadata"]["parent_id"]
        if parent_id not in seen:
            seen.add(parent_id)
            parent_ids.append(parent_id)

    return [parent_store[parent_id] for parent_id in parent_ids[:top_k]]