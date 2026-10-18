from pydantic import BaseModel, Field
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class Citation(BaseModel):
    """A single citation linking a claim to a source document."""

    claim: str = Field(description="The specific statement being cited")
    source_index: int = Field(description="Index of the source document (1-based)")
    source_title: str = Field(description="Title of the cited source")
    quote: str = Field(
        description="Exact or near-exact quote from the source that supports the claim"
    )
    confidence: float = Field(
        description="Confidence that the source supports this claim (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

class CitedResponse(BaseModel):
    """A complete RAG response with structured citations."""

    answer: str = Field(
        description="The full answer with inline citation markers like [1], [2]"
    )
    citations: list[Citation] = Field(
        description="List of citations, one per factual claim"
    )
    unsupported_claims: list[str] = Field(
        description="Claims made that are not supported by any source",
        default_factory=list,
    )
    confidence_score: float = Field(
        description="Overall confidence in the answer (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    answer_completeness: str = Field(
        description="Whether sources fully, partially, or do not answer the question",
        pattern="^(full|partial|insufficient)$",
    )
    
    
    
    
    
# Implementation

load_dotenv()
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

class Citation(BaseModel):
    claim: str = Field(description="The specific statement being cited")
    source_index: int = Field(description="Index of the source document (1-based)")
    source_title: str = Field(description="Title of the cited source")
    quote: str = Field(description="Supporting quote from the source")
    confidence: float = Field(ge=0.0, le=1.0)

class CitedResponse(BaseModel):
    answer: str = Field(description="Answer with inline [N] markers")
    citations: list[Citation]
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    answer_completeness: str

def format_sources(documents: list[dict]) -> str:
    """
    Format retrieved documents as numbered sources for the model.
    Clear numbering helps the model cite accurately.
    """
    formatted = ""
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", f"Document {i}")
        content = doc.get("content", "")
        formatted += f"[{i}] {title}\n{content}\n\n"
    return formatted

def generate_cited_response(
    query: str,
    documents: list[dict],
) -> CitedResponse:
    """
    Generate a response with structured citations using OpenAI's
    structured output feature.
    """
    numbered_sources = format_sources(documents)

    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant that answers questions based on "
                    "provided sources. Follow these rules:\n\n"
                    "1. Only use information from the numbered sources below.\n"
                    "2. For every factual claim, include a citation marker [N] "
                    "where N is the source number.\n"
                    "3. Include the exact quote from the source that supports "
                    "each cited claim.\n"
                    "4. If a claim cannot be supported by any source, list it "
                    "in unsupported_claims.\n"
                    "5. Set confidence_score based on how well the sources "
                    "answer the question (1.0 = strong source coverage, 0.0 = no "
                    "relevant information).\n"
                    "6. Set answer_completeness to 'full' if sources fully "
                    "answer the question, 'partial' if they cover some aspects, "
                    "or 'insufficient' if they barely help.\n"
                    "7. If the sources do not contain enough information to "
                    "answer the question, say so explicitly.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sources:\n{numbered_sources}\n\n"
                    f"Question: {query}\n\n"
                    "Provide your answer with citations."
                ),
            },
        ],
        response_format=CitedResponse,
    )

    return response.choices[0].message.parsed

# Example usage
documents = [
    {
        "title": "HR Policy Manual, Section 4.2",
        "content": (
            "Full-time employees are entitled to twenty (20) days of paid "
            "vacation per calendar year. Vacation days accrue monthly at a "
            "rate of 1.67 days per month."
        ),
    },
    {
        "title": "Employee Handbook, Chapter 7",
        "content": (
            "After three (3) years of continuous service, employees receive "
            "five (5) additional vacation days per year. This tenure bonus "
            "is applied automatically on the employee's work anniversary."
        ),
    },
]

result = generate_cited_response(
    "What is the vacation policy for long-term employees?",
    documents,
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_score}")
print(f"Completeness: {result.answer_completeness}")
print(f"\nCitations:")
for c in result.citations:
    print(f"  [{c.source_index}] {c.claim}")
    print(f"       Quote: \"{c.quote}\"")
    print(f"       Confidence: {c.confidence}")
if result.unsupported_claims:
    print(f"\nUnsupported claims:")
    for claim in result.unsupported_claims:
        print(f"  - {claim}")