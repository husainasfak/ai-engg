from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:  
    raise ValueError("OPENAI_API_KEY is not set")


def generate_answer(
    question: str,
    context: str,
    model: str = "gpt-5.4-mini",
) -> dict:
    """
    Generate an answer using the retrieved context.

    Returns both the answer text and the sources used.
    """
    client = OpenAI(api_key=openai_api_key)

    system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Answer based ONLY on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information to answer, say "I don't have enough information to answer this question."
3. When citing information, reference the source number (e.g., [1], [2]).
4. Keep answers clear and concise.
5. If multiple sources provide relevant information, synthesize them into a coherent answer."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )

    return {
        "answer": response.choices[0].message.content,
        "model": model,
        "tokens_used": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    }