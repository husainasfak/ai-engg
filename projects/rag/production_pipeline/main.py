import importlib.util
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

# Step modules import each other by bare name, so their folders must be on sys.path.
ROOT = Path(__file__).parent
sys.path[:0] = [str(ROOT), str(ROOT / "ingestion_pipeline"), str(ROOT / "query_pipeline")]

from error_handleing import safe_ingest
from incremental_update import incremental_update

# query_pipeline/main.py shares this file's name, so load it by path.
_spec = importlib.util.spec_from_file_location("query_main", ROOT / "query_pipeline" / "main.py")
_query_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_query_main)
query = _query_main.query

load_dotenv()

# --- Ingestion (run once, or when documents change) ---
report = safe_ingest(
    source_dir="./docs",
    collection_name="product_docs",
)
print(f"Ingested {len(report['loaded'])} files, {report['total_chunks']} chunks")

# --- Querying (run per user request) ---
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("product_docs")

# Simple question
result = query("What is the refund policy for digital products?", collection)
print(f"\nQ: What is the refund policy for digital products?")
print(f"A: {result['answer']}")

# Another question. To restrict retrieval to specific sources, thread a
# `where` filter through query() into retrieve() (which already supports it).
result = query(
    "How do I authenticate with the API?",
    collection,
)
print(f"\nQ: How do I authenticate with the API?")
print(f"A: {result['answer']}")

# Check sources
for s in result["sources"]:
    print(f"  Source: {s['source']}, relevance: {s['score']}")

# --- Incremental update (when a single file changes) ---
incremental_update(collection, "docs/api-reference.pdf")