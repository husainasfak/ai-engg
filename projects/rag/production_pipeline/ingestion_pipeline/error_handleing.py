import time
import logging
from pathlib import Path

from openai import OpenAI

from step1_document_loading import load_document
from step2_chunking_and_metadata import chunk_document
from step3_embedding_and_storing import build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embed_with_retry(
    texts: list[str],
    model: str = "text-embedding-3-small",
    max_retries: int = 3,
) -> list[list[float]]:
    """Embed texts with retry logic for API failures."""
    client = OpenAI()

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Embedding failed (attempt {attempt + 1}): {e}. Retrying in {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                raise

def safe_ingest(source_dir: str, collection_name: str = "documents") -> dict:
    """
    Ingest with error tracking. Continues past individual file failures.

    Returns a report of what succeeded and what failed.
    """
    report = {"loaded": [], "failed": [], "total_chunks": 0}

    all_chunks = []
    for file_path in Path(source_dir).rglob("*"):
        if file_path.suffix.lower() in {".pdf", ".md", ".html", ".htm", ".txt"}:
            try:
                docs = load_document(str(file_path))
                for doc in docs:
                    chunks = chunk_document(doc)
                    all_chunks.extend(chunks)
                report["loaded"].append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                report["failed"].append({"file": str(file_path), "error": str(e)})

    report["total_chunks"] = len(all_chunks)

    if all_chunks:
        build_index(all_chunks, collection_name)

    logger.info(
        f"Ingestion complete: {len(report['loaded'])} files loaded, "
        f"{len(report['failed'])} failed, {report['total_chunks']} chunks indexed"
    )
    return report