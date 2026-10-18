# confidence scoring means estimating how well-supported the overall answer is by the retrieved evidence.

"""
4 signals for confidence.
1. Retrieval Quality
First ask:
Did we retrieve relevant documents?

If the retrieved documents barely relate to the question, confidence should be low.
Good retrieval → ↑ confidence
Poor retrieval → ↓ confidence

2. Citation Coverage
Ask:
How many claims in the answer are actually backed by citations?

Example:
5 claims
3 have citations

Citation coverage:
3 / 5 = 60%

That's weaker than:
5 claims
5 have citations

5 / 5 = 100%

3. Grounding Strength
Having a citation isn't enough.
You ask:
How strongly does the source support the claim?

Example:
Strong support:
Claim: “Refunds are available for 30 days.”

Source: “Customers can request refunds within 30 days.”

Weak support:
Claim: “Refunds are available for 30 days.”

Source: “Customers should contact support about refunds.”

The second source is related to refunds but doesn't actually establish the 30-day claim.
4. Source Agreement
If multiple sources are used, check whether they agree.
Source A → 30 days
Source B → 30 days
Source C → 30 days

Good → higher confidence
But:
Source A → 30 days
Source B → 45 days

Now there is a conflict → lower confidence.

"""


from pydantic import BaseModel, Field

class ConfidenceReport(BaseModel):
    """Detailed confidence breakdown for a RAG response."""

    overall_score: float = Field(ge=0.0, le=1.0)
    retrieval_score: float = Field(
        ge=0.0, le=1.0,
        description="How relevant were the retrieved documents",
    )
    citation_coverage: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of claims with citations",
    )
    grounding_score: float = Field(
        ge=0.0, le=1.0,
        description="Average verification score of citations",
    )
    assessment: str = Field(
        description="Short confidence assessment for the product UI",
    )

def compute_confidence(
    cited_response: CitedResponse,
    verification_results: list[dict],
    retrieval_distances: list[float],
) -> ConfidenceReport:
    """
    Compute a confidence score from multiple signals.
    """
    # Signal 1: Retrieval quality
    # Convert distances to similarities (lower distance = higher similarity)
    if retrieval_distances:
        similarities = [1 / (1 + d) for d in retrieval_distances]
        retrieval_score = sum(similarities) / len(similarities)
    else:
        retrieval_score = 0.0

    # Signal 2: Citation coverage
    total_claims = len(cited_response.citations) + len(
        cited_response.unsupported_claims
    )
    if total_claims > 0:
        citation_coverage = len(cited_response.citations) / total_claims
    else:
        citation_coverage = 0.0

    # Signal 3: Grounding strength
    if verification_results:
        grounded_count = sum(
            1 for v in verification_results if v.get("is_grounded", False)
        )
        grounding_score = grounded_count / len(verification_results)
    else:
        grounding_score = 0.0

    # Overall score: weighted combination
    overall = (
        0.3 * retrieval_score
        + 0.3 * citation_coverage
        + 0.4 * grounding_score
    )

    # Short assessment for the product UI
    if overall >= 0.8:
        assessment = "High confidence. The answer is well supported by the retrieved sources."
    elif overall >= 0.5:
        assessment = "Moderate confidence. Some claims may need review."
    elif overall >= 0.3:
        assessment = "Low confidence. Many claims lack clear source support."
    else:
        assessment = "Very low confidence. Do not show without review."

    return ConfidenceReport(
        overall_score=round(overall, 2),
        retrieval_score=round(retrieval_score, 2),
        citation_coverage=round(citation_coverage, 2),
        grounding_score=round(grounding_score, 2),
        assessment=assessment,
    )