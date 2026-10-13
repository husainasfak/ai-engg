from openai import OpenAI
from dotenv import load_dotenv
import os 
from sentence_transformers import SentenceTransformer

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(api_key=openai_api_key)
#  open ai embedding modal
response = client.embeddings.create(
    model="text-embedding-3-small",
    
    input="How do I fix a broken pipe?"
)

# Extract the embedding vector
embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")


# If you prefer not to depend on an external API, the sentence-transformers library gives you access to many local and open-weight embedding models.
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "How do I fix a broken pipe?",
    "Plumbing repair guide",
    "Chocolate cake recipe"
]
embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")
print(f"First embedding (first 10 values): {embeddings[0][:10]}")


# Practical Considerations for Embeddings
Dimensionality = """
Higher dimensions mean more storage and potentially slower searches, but not necessarily better results. Choose dimensions based on retrieval quality and cost. OpenAI's text-embedding-3 models support reducing dimensions.
"""

Normalization ="""
Scales a vector to length 1. With normalized vectors, cosine similarity and dot product produce the same ranking. Check whether your model already normalizes its output.
Do not assume every provider or local model normalizes outputs. If your retrieval setup expects cosine-like behavior, normalize explicitly unless the model documentation says the vectors are already unit length:
"""
import numpy as np

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Normalize a batch of embeddings
embeddings_normalized = np.array([normalize(e) for e in embeddings])

# Verify: magnitude should be ~1.0
magnitudes = np.linalg.norm(embeddings_normalized, axis=1)
print(f"Magnitudes: {magnitudes}")


batch_processing="""
Process large volumes of text in batches instead of one request per text. Respect API rate limits, token limits, and memory constraints. Use bounded retries for temporary failures.

For API-based embedding, the main constraints are:
- Maximum number of inputs per request
- Maximum tokens per input and per request
- Requests-per-minute and tokens-per-minute rate limits
- Retry behavior for transient provider errors
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed a large dataset in batches
large_dataset = [f"Document number {i} about various topics" for i in range(10000)]

# sentence-transformers handles batching internally
all_embeddings = model.encode(
    large_dataset,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True  # Normalize during encoding
)

print(f"Embedded {len(all_embeddings)} documents")
print(f"Shape: {all_embeddings.shape}")

import time
from openai import OpenAI, RateLimitError, APIError

client = OpenAI()

def embed_in_batches(texts, model="text-embedding-3-small", batch_size=100, max_retries=4):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, input=batch)
                all_embeddings.extend(item.embedding for item in response.data)
                print(f"Processed {i + len(batch)}/{len(texts)}")
                break
            except (RateLimitError, APIError):
                if attempt == max_retries - 1:
                    raise

                wait_seconds = 2 ** attempt
                time.sleep(wait_seconds)

    return all_embeddings



Embedding_Pipeline =  """The typical workflow is:
Clean text → Split into chunks → Batch → Generate embeddings → Normalize if needed → Store vectors with metadata."""


Caching_Embeddings = """
Don't regenerate embeddings for unchanged text. Cache them using a text hash and track the embedding model, dimensions, normalization settings, and chunking version.
"""
"""
Embeddings are useful because learned geometry makes comparison possible. Training shapes that geometry, and pooling produces the single vector used for search, clustering, or ranking
"""


# Evaluation Metrics

# Two metrics are especially useful for embedding evaluation:

# Mean Reciprocal Rank (MRR) measures where the first relevant document appears in your ranked results. If the correct document is ranked first, the reciprocal rank is 1. If it is ranked third, the reciprocal rank is 1/3. MRR is the average across all evaluated queries.

# Recall@K measures how often a relevant document appears anywhere in the top K results. Recall@10 asks: "Is useful evidence somewhere in the top 10?" This is important for RAG systems because the model can only use documents that retrieval actually returns.

# Building a Benchmarking Pipeline
import time
import numpy as np
from sentence_transformers import SentenceTransformer

def compute_cosine_similarity(query_embedding, doc_embeddings):
    """Compute cosine similarity between a query and all documents."""
    # With normalized embeddings, cosine similarity = dot product
    return np.dot(doc_embeddings, query_embedding)

def evaluate_model(model_name, queries, documents, relevant_doc_indices, k=10):
    """
    Evaluate a single embedding model on a retrieval task.

    Args:
        model_name: HuggingFace model name
        queries: List of query strings
        documents: List of document strings
        relevant_doc_indices: List of ints. relevant_doc_indices[i] is the
        index of a relevant document for queries[i]
        k: Number of top results to consider for recall

    Returns:
        Dictionary with MRR, recall@k, and latency metrics
    """
    model = SentenceTransformer(model_name)

    # Measure embedding latency
    start_time = time.time()
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    doc_embed_time = time.time() - start_time

    start_time = time.time()
    query_embeddings = model.encode(queries, normalize_embeddings=True)
    query_embed_time = time.time() - start_time

    # Compute retrieval metrics.
    reciprocal_ranks = []
    recall_at_k = 0

    for i, query_emb in enumerate(query_embeddings):
        similarities = compute_cosine_similarity(query_emb, doc_embeddings)
        ranked_indices = np.argsort(similarities)[::-1]  # Descending order

        relevant_doc_idx = relevant_doc_indices[i]
        rank = np.where(ranked_indices == relevant_doc_idx)[0][0] + 1

        reciprocal_ranks.append(1.0 / rank)
        if rank <= k:
            recall_at_k += 1

    mrr = np.mean(reciprocal_ranks)
    recall = recall_at_k / len(queries)

    return {
        "model": model_name,
        "mrr": round(mrr, 4),
        f"recall@{k}": round(recall, 4),
        "doc_embed_time_sec": round(doc_embed_time, 2),
        "query_embed_time_sec": round(query_embed_time, 2),
        "dimensions": doc_embeddings.shape[1],
    }
    

# Example: benchmark 3 models
queries = [
    "How does gradient descent work?",
    "What is a transformer model?",
    "Explain backpropagation",
]

documents = [
    "Gradient descent is an optimization algorithm that iteratively adjusts parameters...",
    "The transformer architecture uses self-attention mechanisms to process sequences...",
    "Backpropagation computes gradients of the loss function with respect to weights...",
    "Random forests combine multiple decision trees for improved accuracy...",
    "Convolutional neural networks use filters to extract spatial features from images...",
]

# relevant_doc_indices[i] = index of a relevant document for queries[i]
relevant_doc_indices = [0, 1, 2]

models_to_test = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
]

results = []
for model_name in models_to_test:
    print(f"Evaluating {model_name}...")
    result = evaluate_model(model_name, queries, documents, relevant_doc_indices)
    results.append(result)
    print(f"  MRR: {result['mrr']}, Recall@10: {result['recall@10']}")
    print(f"  Dimensions: {result['dimensions']}")
    print(f"  Doc embed time: {result['doc_embed_time_sec']}s")
    print()
    
    
"""
Embedding model choice is an infrastructure decision. Retrieval score estimates relevance, while model size, dimensions, token limits, and model-specific conventions determine practical production behavior.
"""
