# This code implements a hybrid search engine with reranking. It combines three techniques:
# 1. Vector search — finds text with similar meaning.
# 2. BM25 keyword search — finds text containing matching keywords.
# 3. Cross-encoder reranking — reorders the retrieved passages based on their relevance to the user's question.
# Architecture:
"""
1. Documents
2. Chunking - split documents into smaller passages
3. Embedding - convert passages into vector representations and Tokenize passages for BM25 keyword index
4. User submits a query - Both search methods retrieve candidates
5. Reciprocal Rank Fusion (RRF) - Combine the ranked lists
6. Cross-encoder reranking - Score candidates against the query
7. Final ranked results - Return text, metadata, and scores


"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np
import re

class SemanticSearchEngine:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        # Validating the overlap
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Loading the models
        # These models have different jobs. The embedding model helps you search a large collection efficiently. The cross-encoder helps you choose better results from a smaller collection of candidates.
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(rerank_model)
        
        # Initializing ChromaDB
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            "search_engine"
        )
        
        self.chunks = [] # The text of every chunk.
        self.chunk_meta = [] # The metadata corresponding to each chunk.
        self.bm25 = None # The keyword index, initially unavailable because no documents have been ingested.
        
    # preparing documents for search
    def ingest(self, documents: list[dict]):
        """
        Ingest documents into the search engine.
        Each document: {"text": str, "source": str, "title": str}
        """
        start = len(self.chunks)
        for doc in documents:
            doc_chunks = self._chunk_document(doc["text"])
            for chunk_text in doc_chunks:
                self.chunks.append(chunk_text)
                self.chunk_meta.append({
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                })

        # Build the BM25 index
        tokenized = [self._tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Build the vector index This converts every chunk into a vector.
        new_chunks = self.chunks[start:]
        if not new_chunks:
            return
        embeddings = self.embedder.encode(new_chunks, show_progress_bar=True)
        # store the chunks in ChromaDB.
        self.collection.add(
            ids=[str(i) for i in range(start, len(self.chunks))],
            embeddings=embeddings.tolist(), # .tolist() converts the NumPy array into ordinary Python lists, which ChromaDB accepts.
            documents=new_chunks,
        )
        print(
            f"Ingested {len(documents)} documents "
            f"into {len(self.chunks)} chunks"
        )
    
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 50,
        use_reranking: bool = True,
    ) -> list[dict]:
        """
            Search with hybrid retrieval and optional re-ranking.
        """
        if self.bm25 is None:
            return []
        # Stage 1: hybrid retrieval
        bm25_ranked = self._bm25_search(query, top_k=retrieval_k)
        semantic_ranked = self._semantic_search(query, top_k=retrieval_k)
        # The RRF function combines the two ranked lists and gives chunks that rank well across both methods an advantage.
        fused = reciprocal_rank_fusion([bm25_ranked, semantic_ranked])
        candidate_ids = [doc_id for doc_id, _ in fused[:retrieval_k]]

        # Stage 2: optional  reranking
        if use_reranking and candidate_ids:
            candidate_texts = [self.chunks[i] for i in candidate_ids]
            pairs = [[query, text] for text in candidate_texts]
            scores = self.reranker.predict(pairs)

            scored = list(zip(candidate_ids, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            final_ids = scored[:top_k]
        else:
            final_ids = [(doc_id, score) for doc_id, score in fused[:top_k]]

        # Format the results
        results = []
        for doc_id, score in final_ids:
            results.append({
                "chunk_id": doc_id,
                "text": self.chunks[doc_id],
                "score": float(score),
                "metadata": self.chunk_meta[doc_id],
            })

        return results

    def _chunk_document(self, text: str) -> list[str]:
        """Recursive chunking with paragraph and sentence boundaries."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                for part in parts:
                    candidate = current + sep + part if current else part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        if len(part) > self.chunk_size:
                            remaining = separators[separators.index(sep) + 1:]
                            chunks.extend(self._recursive_split(part, remaining))
                            current = ""
                        else:
                            current = part
                if current:
                    chunks.append(current)
                return chunks

        # Fallback: fixed-size splitting
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [int(i) for i in top_indices if scores[i] > 0]

    def _semantic_search(self, query: str, top_k: int) -> list[int]:
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
        )
        return [int(id) for id in results["ids"][0]]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists: score(d) = sum(1 / (k + rank)). Returns [(doc_id, score)] best first."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    assert reciprocal_rank_fusion([[1, 2], [2, 3]])[0][0] == 2
    engine = SemanticSearchEngine(chunk_size=100, chunk_overlap=10)
    engine.ingest([
        {"text": "Python is a programming language. " * 10, "source": "a", "title": "Python"},
        {"text": "Cats are small furry animals that purr.", "source": "b", "title": "Cats"},
    ])
    results = engine.search("furry pets", top_k=2)
    print(results)
    assert results[0]["metadata"]["title"] == "Cats"







def evaluate_search(
    search_fn,
    queries: list[str],
    ground_truth: dict[str, list[int]],
    k: int = 10,
) -> dict:
    """
    Evaluate search quality.

    Args:
        search_fn: Function that takes query and returns list of doc IDs
        queries: List of test queries
        ground_truth: {query: [relevant_doc_ids]}
        k: Number of results to evaluate
    """
    mrr_total = 0.0
    recall_total = 0.0
    precision_total = 0.0
    evaluated = 0

    for query in queries:
        results = search_fn(query)[:k]
        relevant = set(ground_truth.get(query, []))

        if not relevant:
            continue

        evaluated += 1

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, doc_id in enumerate(results, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        mrr_total += rr

        # Recall@k
        found = len(set(results) & relevant)
        recall_total += found / len(relevant)

        # Precision@k
        precision_total += found / k

    if evaluated == 0:
        return {"MRR": 0.0, "Recall@k": 0.0, "Precision@k": 0.0}

    return {
        "MRR": mrr_total / evaluated,
        "Recall@k": recall_total / evaluated,
        "Precision@k": precision_total / evaluated,
    }