from dataclasses import dataclass, field
from xml.dom.minidom import Document


@dataclass
class Chunk:
    """A chunk of text with full metadata for retrieval and citation."""
    content: str
    metadata: dict = field(default_factory=dict)

def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split a document into chunks, preserving metadata."""
    text = document.content
    separators = ["\n\n", "\n", ". ", " "]

    chunks = recursive_split(text, chunk_size, separators)

    # Apply overlap once, after the text is fully split
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = add_overlap(chunks, chunk_overlap)

    return [
        Chunk(
            content=chunk_text,
            metadata={
                **document.metadata,  # Inherit all document metadata
                "chunk_index": i,
                "chunk_total": len(chunks),
                # Approximate offset; not exact once overlap is prepended
                "char_start": text.find(chunk_text[:50]),
            }
        )
        for i, chunk_text in enumerate(chunks)
    ]

def recursive_split(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text using progressively finer separators."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Use the first separator that appears in the text
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            finer = separators[i + 1:]  # Finer separators for oversized parts
            chunks = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                    continue

                # Flush what we have accumulated so far
                if current:
                    chunks.append(current.strip())
                    current = ""

                # A single part can still exceed chunk_size: split it further
                if len(part) > chunk_size:
                    chunks.extend(recursive_split(part, chunk_size, finer))
                else:
                    current = part

            if current:
                chunks.append(current.strip())

            return [c for c in chunks if c.strip()]

    # No separator works: force a hard split at chunk_size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of each chunk to the start of the next."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + " " + chunks[i])
    return result