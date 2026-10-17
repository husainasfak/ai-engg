# ranking means taking the documents retrieved by the first search and then re-ordering them based on how relevant they actually are to the user's question.
"""
First-stage vector/hybrid search finds 5 documents:
1. Refund eligibility depends on product category.
2. Shipping refunds are for lost parcels.
3. Refunds are credited to the original card.
4. Orders can be returned within 30 days.
5. Store credit never expires.

The first search isn't necessarily perfect.
A reranker reads the question + each candidate document together and decides which is most relevant.
It might reorder them:
1. Orders can be returned within 30 days. ← most relevant
2. Refund eligibility depends on product category.
3. Refunds are credited to the original card.
4. Shipping refunds are for lost parcels.
5. Store credit never expires.

Why is this useful?
The first search is optimized to retrieve broadly—for example, the top 20–100 candidates.
The reranker is optimized to choose precisely from those candidates.    Pasted text
Think of it as:
Retrieve → Rerank → Give best results to LLM
Rank fusion vs reranking
Don't confuse them:
- Rank fusion: combines rankings from different search systems.
- Reranking: takes the retrieved candidates and re-evaluates their relevance to the original question.
So:
Hybrid search → Rank fusion → Reranking → Final context → LLM

"""


from pydantic import BaseModel, Field

class RerankScore(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    reason: str

def rerank_with_llm(question: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Simple LLM reranker. Use a dedicated reranker for high-volume systems."""
    scored = []

    for candidate in candidates:
        response = client.chat.completions.parse(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Score how relevant this passage is for answering the question. "
                        "Score 1.0 for directly answerable, 0.0 for unrelated. "
                        "Be strict about scope, dates, product names, and negation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Passage:\n{candidate['content']}"
                    ),
                },
            ],
            response_format=RerankScore,
            temperature=0,
        )

        score = response.choices[0].message.parsed
        scored.append({**candidate, "rerank_score": score.relevance})

    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


