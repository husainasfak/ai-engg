import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

def trusted_rag_pipeline(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 5,
    confidence_threshold: float = 0.4,
) -> dict:
    """
    Complete RAG pipeline with citations, verification, and confidence.
    """
    # Step 1: Retrieve documents. Embed the query with the same model used at
    # ingestion, then search by vector. Passing query_texts instead would make
    # ChromaDB use its default embedder, which would not match the index.
    q_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "title": results["metadatas"][0][i].get("title", f"Doc {i+1}"),
            "content": results["documents"][0][i],
        })

    # Step 2: Generate response with structured citations
    cited_response = generate_cited_response(query, documents)

    # Step 3: Verify each citation
    verification_results = []
    for citation in cited_response.citations:
        source_idx = citation.source_index - 1
        if 0 <= source_idx < len(documents):
            result = verify_citation_hybrid(
                claim=citation.claim,
                source_text=documents[source_idx]["content"],
            )
            verification_results.append(result)

    # Step 4: Compute confidence
    confidence = compute_confidence(
        cited_response,
        verification_results,
        results["distances"][0],
    )

    # Step 5: Check faithfulness of the full response
    full_context = "\n\n".join(doc["content"] for doc in documents)
    faithfulness = check_faithfulness(cited_response.answer, full_context)

    # Step 6: Decide whether to show the response
    abstain, reason = should_abstain(cited_response)

    if abstain:
        return {
            "status": "abstained",
            "reason": reason,
            "confidence": confidence.model_dump(),
        }

    return {
        "status": "answered",
        "answer": cited_response.answer,
        "citations": [c.model_dump() for c in cited_response.citations],
        "unsupported_claims": cited_response.unsupported_claims,
        "confidence": confidence.model_dump(),
        "faithfulness": {
            "score": faithfulness.faithfulness_score,
            "unsupported_claims": faithfulness.unsupported_claims,
        },
    }