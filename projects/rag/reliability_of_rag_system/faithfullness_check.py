
"""
Faithfulness means the response stays within the provided context. An unfaithful response contains unsupported claims: statements that may sound correct but are not grounded in the source documents.


Faithfulness checking is different from grounding verification. Grounding verification checks individual citations: does source [1] support this claim? Faithfulness checking evaluates the entire response against all provided context: does anything in the response go beyond what the sources say?
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

class AtomicClaims(BaseModel):
    """List of atomic factual claims extracted from a response."""

    claims: list[str] = Field(
        description="Each atomic factual claim as a standalone statement"
    )

class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""

    claim: str
    is_supported: bool = Field(
        description="Whether the context supports this claim"
    )
    supporting_evidence: str = Field(
        description="Quote from context that supports or contradicts the claim, "
        "or 'No relevant evidence found'"
    )
    verdict: str = Field(
        description="'supported', 'contradicted', or 'not_found'"
    )

class FaithfulnessReport(BaseModel):
    """Complete faithfulness analysis of a response."""

    verdicts: list[ClaimVerdict]
    faithfulness_score: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of claims supported by context",
    )
    unsupported_claims: list[str] = Field(
        description="Claims not supported by context"
    )

def extract_atomic_claims(response_text: str) -> list[str]:
    """
    Break a response into atomic, independently verifiable claims.
    """
    result = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract every factual claim from the text below. Each claim "
                    "should be:\n"
                    "- Atomic: contains exactly one piece of information\n"
                    "- Self-contained: understandable without the surrounding text\n"
                    "- Factual: a verifiable statement, not an opinion\n\n"
                    "Do NOT include meta-statements like 'according to the document' "
                    "or 'the text says'. Just state the facts."
                ),
            },
            {"role": "user", "content": response_text},
        ],
        response_format=AtomicClaims,
    )

    return result.choices[0].message.parsed.claims

def check_claim_against_context(
    claim: str,
    context: str,
) -> ClaimVerdict:
    """
    Check whether a single claim is supported by the provided context.
    """
    result = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checker. Given a claim and a context, determine "
                    "if the context supports the claim.\n\n"
                    "Rules:\n"
                    "- 'supported': The context directly states or clearly implies "
                    "the claim\n"
                    "- 'contradicted': The context says something that conflicts "
                    "with the claim\n"
                    "- 'not_found': The context does not contain information "
                    "relevant to the claim\n\n"
                    "Provide the exact quote from the context that relates to the "
                    "claim, or state 'No relevant evidence found'."
                ),
            },
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nContext: {context}",
            },
        ],
        response_format=ClaimVerdict,
    )

    return result.choices[0].message.parsed

def check_faithfulness(
    response_text: str,
    context: str,
) -> FaithfulnessReport:
    """
    Full faithfulness check: extract claims, verify each one,
    compute overall score.
    """
    # Step 1: Extract atomic claims
    claims = extract_atomic_claims(response_text)
    print(f"Extracted {len(claims)} claims to verify")

    # Step 2: Verify each claim
    verdicts = []
    for claim in claims:
        verdict = check_claim_against_context(claim, context)
        verdicts.append(verdict)

    # Step 3: Compute faithfulness score
    supported_count = sum(1 for v in verdicts if v.verdict == "supported")
    faithfulness_score = supported_count / len(verdicts) if verdicts else 0.0

    unsupported = [
        v.claim for v in verdicts if v.verdict in ("not_found", "contradicted")
    ]

    return FaithfulnessReport(
        verdicts=verdicts,
        faithfulness_score=round(faithfulness_score, 2),
        unsupported_claims=unsupported,
    )