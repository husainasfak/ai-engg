# grounding means making sure the model's answer is supported by the information retrieved from the source documents.
# Approach 1: Embedding Similarity
# The fastest approach is to embed the claim and the cited source text, then compute cosine similarity. High similarity can suggest the source is related to the claim, but it does not prove support.

# Problem - Embeddings measure semantic similarity, not logical correctness.

from sentence_transformers import SentenceTransformer, util

def verify_citation_embedding(
    claim: str,
    source_text: str,
    threshold: float = 0.5,
) -> dict:
    """
    Verify a citation using embedding similarity.
    Fast but can miss subtle mismatches.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    claim_embedding = model.encode(claim, convert_to_tensor=True)
    source_embedding = model.encode(source_text, convert_to_tensor=True)

    similarity = util.cos_sim(claim_embedding, source_embedding).item()

    return {
        "claim": claim,
        "similarity": round(similarity, 3),
        "is_grounded": similarity >= threshold,
        "method": "embedding_similarity",
    }
    

# Approach 2: Judge Model
# Give the claim + source to another LLM and ask: “Does this source actually support this claim?”
"""
Numerical mismatches. "The refund window is 30 days" cited against a source that says "45-day return policy." The embeddings are very similar, but the facts differ.

Negation errors. "Employees are eligible for the bonus" cited against a source that says "Employees in their first year are not eligible for the bonus." Again, high embedding similarity, but opposite meaning.

Scope mismatches. "All employees receive 20 vacation days" cited against a source that says "Full-time employees receive 20 vacation days." The claim is broader than what the source supports.
"""


from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

class GroundingVerdict(BaseModel):
    """Structured output for grounding verification."""

    is_supported: bool = Field(
        description="Whether the source supports the claim"
    )
    reasoning: str = Field(
        description="Brief explanation of the verdict"
    )
    support_type: str = Field(
        description="Type of support: 'direct', 'inferred', 'partial', 'contradicted', 'unrelated'",
    )

def verify_citation_llm(claim: str, source_text: str) -> GroundingVerdict:
    """
    Use a model to verify whether a source supports a claim.
    More accurate than embedding similarity, especially for
    numerical claims and subtle distinctions.
    """
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. Given a claim and a "
                    "source text, determine if the source supports the claim.\n\n"
                    "Support types:\n"
                    "- 'direct': Source explicitly states the claim\n"
                    "- 'inferred': Claim can be reasonably inferred from source\n"
                    "- 'partial': Source supports part of the claim but not all\n"
                    "- 'contradicted': Source contradicts the claim\n"
                    "- 'unrelated': Source has nothing to do with the claim\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Claim: {claim}\n\n"
                    f"Source text: {source_text}\n\n"
                    "Does the source support this claim?"
                ),
            },
        ],
        response_format=GroundingVerdict,
    )

    return response.choices[0].message.parsed


# Approach 3: Combining Both
# This hybrid approach gives you speed on clear cases and deeper review where similarity is ambiguous. The split between "clear" and "needs judgment" is corpus-dependent, so calibrate thresholds on labeled examples instead of copying them blindly.


def verify_citation_hybrid(
    claim: str,
    source_text: str,
    similarity_threshold_high: float = 0.75,
    similarity_threshold_low: float = 0.3,
) -> dict:
    """
    Two-stage verification: fast embedding check, then model review for edge cases.
    """
    # Stage 1: Fast embedding check
    embedding_result = verify_citation_embedding(claim, source_text)

    # Clear match: skip model review
    if embedding_result["similarity"] >= similarity_threshold_high:
        return {
            "is_grounded": True,
            "method": "embedding_high_confidence",
            "similarity": embedding_result["similarity"],
        }

    # Clear mismatch: skip model review
    if embedding_result["similarity"] < similarity_threshold_low:
        return {
            "is_grounded": False,
            "method": "embedding_low_confidence",
            "similarity": embedding_result["similarity"],
        }

    # Uncertain: use a model for detailed verification
    llm_result = verify_citation_llm(claim, source_text)

    return {
        "is_grounded": llm_result.is_supported,
        "method": "llm_verification",
        "similarity": embedding_result["similarity"],
        "support_type": llm_result.support_type,
        "reasoning": llm_result.reasoning,
    }