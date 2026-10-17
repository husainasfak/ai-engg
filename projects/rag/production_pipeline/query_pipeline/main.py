import chromadb
from openai import OpenAI

from step_1_query_preprocessing import preprocess_query
from step_2_retrieval import retrieve
from step3_context_assembly import assemble_context
from step4_answer_generation import generate_answer


def query(
    question: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    use_preprocessing: bool = True,
    model: str = "gpt-5.4-mini",
) -> dict:
    """
    Full query pipeline: preprocess, retrieve, assemble, generate.

    Args:
        question: The user's question
        collection: ChromaDB collection to search
        n_results: Number of chunks to retrieve
        use_preprocessing: Whether to use model-based query preprocessing
        model: Model used for generation

    Returns:
        Dict with 'answer', 'sources', 'model', and 'tokens_used'
    """
    client = OpenAI() if use_preprocessing else None

    # Step 1: Preprocess the query
    search_query = preprocess_query(question, client)

    # Step 2: Retrieve relevant chunks
    retrieved = retrieve(search_query, collection, n_results)

    # Step 3: Assemble context with citations
    context = assemble_context(retrieved)

    # Step 4: Generate the answer
    result = generate_answer(question, context, model)

    # Attach source information to the result
    result["sources"] = [
        {
            "source": chunk["metadata"].get("source", "unknown"),
            "page": chunk["metadata"].get("page"),
            "section": chunk["metadata"].get("section"),
            "score": round(chunk["score"], 3),
        }
        for chunk in retrieved
    ]
    result["search_query"] = search_query

    return result





# Load the collection (assumes you already ran the ingestion pipeline)
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("product_docs")

# Ask a question
result = query("What's the return policy for electronics?", collection)

print(f"Answer: {result['answer']}\n")
print("Sources:")
for source in result["sources"]:
    page = f", page {source['page']}" if source["page"] else ""
    section = f", section: {source['section']}" if source["section"] else ""
    print(f"  - {source['source']}{page}{section} (relevance: {source['score']})")