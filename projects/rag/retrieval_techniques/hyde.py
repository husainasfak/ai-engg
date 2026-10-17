# Hypothetical Document Embeddings (HyDE) is a technique that generates hypothetical documents based on a given query. It uses a language model to create these documents, which can then be used to improve retrieval performance in information retrieval systems. The generated documents are designed to be relevant to the query, providing additional context and information that may not be present in the original documents.

# User query: "How do refunds work for downloads?"

# Hypothetical passage: "Digital downloads are eligible for refund within a limited window after purchase unless the file has been accessed or consumed. Refunds are processed back to the original payment method."

def generate_hypothetical_document(question: str) -> str:
    """Generate a plausible answer-shaped passage for retrieval."""
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a concise hypothetical document passage that would answer "
                    "the user's question. It is only for search, so use likely domain "
                    "terms. Do not include citations."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=180,
    )
    return response.choices[0].message.content.strip()

def hyde_retrieve(question: str, collection, embedding_model: str, top_k: int = 5):
    """Retrieve using an embedding of a hypothetical answer."""
    hypothetical = generate_hypothetical_document(question)

    embedding = client.embeddings.create(
        model=embedding_model,
        input=hypothetical,
    ).data[0].embedding

    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    

# HyDE can help with short, vague, or abstract questions. It is usually a poor fit for exact identifiers, legal wording, API fields, or anything where invented phrasing might pull retrieval toward the wrong topic. Evaluate it against a plain vector-search baseline before using it broadly.