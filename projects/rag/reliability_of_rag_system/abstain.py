SYSTEM_PROMPT = """You answer questions using only the provided sources.
Follow these rules:

1. If the sources contain enough information to answer the question fully,
   provide a complete answer with citations.

2. If the sources contain some relevant information but not enough for a
   complete answer, provide what you can and explicitly state what is missing.
   Set answer_completeness to "partial".

3. If the sources contain NO relevant information, respond with:
   "I don't have enough information in the available sources to answer
   this question."
   Set answer_completeness to "insufficient" and confidence_score to 0.1.

4. Do not use general knowledge to fill gaps. If it is not in the
   sources, it does not exist for the purpose of this answer.

5. It is better to say "I don't know" than to guess."""

def generate_with_abstention(
    query: str,
    documents: list[dict],
) -> CitedResponse:
    """
    Generate a response that will abstain when sources are insufficient.
    """
    numbered_sources = format_sources(documents)

    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Sources:\n{numbered_sources}\n\n"
                    f"Question: {query}\n\n"
                    "Answer based only on the sources provided."
                ),
            },
        ],
        response_format=CitedResponse,
    )

    return response.choices[0].message.parsed





def should_abstain(
    cited_response: CitedResponse,
    min_confidence: float = 0.3,
    min_citation_ratio: float = 0.5,
) -> tuple[bool, str]:
    """
    Check whether the system should abstain from showing this response.
    Returns (should_abstain, reason).
    """
    # Check 1: Model's own confidence assessment
    if cited_response.answer_completeness == "insufficient":
        return True, "Model reported insufficient source coverage"

    # Check 2: Confidence score too low
    if cited_response.confidence_score < min_confidence:
        return True, f"Confidence score ({cited_response.confidence_score}) below threshold ({min_confidence})"

    # Check 3: Too many unsupported claims
    total_claims = len(cited_response.citations) + len(
        cited_response.unsupported_claims
    )
    if total_claims > 0:
        citation_ratio = len(cited_response.citations) / total_claims
        if citation_ratio < min_citation_ratio:
            return True, (
                f"Only {citation_ratio:.0%} of claims are cited "
                f"(threshold: {min_citation_ratio:.0%})"
            )

    return False, "Response passes all checks"

def safe_rag_response(
    query: str,
    documents: list[dict],
) -> dict:
    """
    Full RAG pipeline with abstention safety net.
    """
    # Generate response with citations
    cited_response = generate_with_abstention(query, documents)

    # Check whether the response should be shown
    abstain, reason = should_abstain(cited_response)

    if abstain:
        return {
            "answer": (
                "I don't have enough information in the available sources "
                "to confidently answer this question."
            ),
            "abstention_reason": reason,
            "confidence": cited_response.confidence_score,
            "partial_info": (
                cited_response.answer
                if cited_response.answer_completeness == "partial"
                else None
            ),
        }

    return {
        "answer": cited_response.answer,
        "citations": [c.model_dump() for c in cited_response.citations],
        "confidence": cited_response.confidence_score,
        "completeness": cited_response.answer_completeness,
    }