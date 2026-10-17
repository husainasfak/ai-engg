
from instructor.auto_client import Optional
from openai import OpenAI


def preprocess_query(query: str, client: Optional[OpenAI] = None) -> str:
    """
    Clean and optionally expand the query for better retrieval.

    For simple queries, basic cleaning is enough.
    For complex or ambiguous queries, use the model to reformulate.
    """
    # Basic cleaning
    cleaned = query.strip()
    cleaned = " ".join(cleaned.split())  # Normalize whitespace

    # For short or simple queries, basic cleaning is sufficient
    if len(cleaned.split()) <= 10 and client is None:
        return cleaned

    # For complex queries, use the model to reformulate
    if client:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's question as a clear, specific search query. "
                        "Keep the core intent. Fix grammar and spelling. "
                        "If the question has multiple parts, focus on the main question. "
                        "Return only the rewritten query, nothing else."
                    ),
                },
                {"role": "user", "content": cleaned},
            ],
            temperature=0,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    return cleaned