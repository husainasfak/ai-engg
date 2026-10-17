# Semantic + keyword search
# Vector search and keyword search are good at different things.
def reciprocal_rank_fusion(result_lists: list[list[dict]], top_k: int = 10, k: int = 60):
    """
    Merge ranked result lists using reciprocal rank fusion.
    This avoids comparing raw scores from different retrieval systems.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            by_id[doc_id] = item

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [by_id[doc_id] for doc_id in ranked_ids[:top_k]]

def hybrid_search(question: str, vector_search, keyword_search, top_k: int = 10):
    vector_results = vector_search(question, top_k=top_k)
    keyword_results = keyword_search(question, top_k=top_k)
    return reciprocal_rank_fusion([vector_results, keyword_results], top_k=top_k)