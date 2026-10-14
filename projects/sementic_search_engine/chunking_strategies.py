#  Strategy 1: Fixed-Size Chunking
# split the documentes into chunks of a fixed number of charcaters or tokens

def fixed_size_chunk(text:str, chunk_size: int = 500, overlap:int = 50) -> list:
    """
    Split text into fixed size chunks with overlap.
    Fixed-size chunking is fast and predictable. It is also blunt. It can cut through sentences, paragraphs, code blocks, and tables without noticing the boundary. A sentence about "the benefits of caching" might be split across two chunks, with neither chunk carrying the full idea.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap # step back by overlaping amount
    return chunk

# Example
document = """
It was a rainy Friday evening when Aarav left his office. He was tired, hungry, and frustrated. He had spent the entire day fixing bugs in a project that seemed to have no end. All he wanted was to reach home, eat something, and sleep.
When he arrived at the railway station, his usual train had already left. He checked his phone and saw that the next train would arrive in forty minutes.
“Perfect,” he whispered. “Just what I needed.”
He sat on a cold metal bench and watched the rain fall on the empty tracks. The station was quieter than usual. A few people stood near the tea stall, and a stray dog slept beneath a wooden table.
After a few minutes, an elderly man wearing a faded blue uniform came and sat at the other end of the bench. He was a station cleaner. His clothes were slightly wet, and he carried a small bag.
“You look upset, young man,” the old man said with a smile.
Aarav laughed softly. “Just a bad day at work.”
The old man nodded. “Work can make us tired. But sometimes, we make ourselves tired by thinking that work is everything.”
Aarav looked at him with curiosity.
The old man continued, “I spent thirty years working here. Every day, I watched people running to catch trains. Some were going to work, some were returning home, and some were leaving their families behind. Everyone seemed to be in a hurry.”
He paused and looked at the tracks.
“I was always in a hurry too. I wanted to earn more money and give my daughter a better life. I missed her school performances, her birthdays, and many small moments. I kept telling myself that I would make time for her later.”
“What happened?” Aarav asked.
“My daughter moved to another city. We slowly stopped talking. One day, I wrote her a letter, but I never sent it. I thought I would send it when I found the right words.”
The old man smiled sadly.
“Years passed. Now, I don't even know if she still lives at the same address.”
Aarav didn't know what to say. The sound of rain filled the silence between them.
Suddenly, he remembered his father.
For the past few months, his father had called him almost every evening. Sometimes Aarav answered, but most of the time, he rejected the calls because he was busy. He always told himself that he would call back after finishing his work.
But work never really finished.
His father never complained. He simply said, “It's okay, beta. Call me whenever you have time.”
Aarav looked at his phone. There were two missed calls from home.
For a moment, he hesitated. He had several unread messages and a few tasks waiting for him. He could call tomorrow, he thought.
Then he remembered the old man's unfinished letter.
Aarav pressed the call button.
His father answered after two rings.
“Hello, beta! Everything okay?”
Aarav smiled. “Yes, Papa. Everything is fine. I just wanted to talk to you.”
There was a brief silence on the other end.
“Really? You called just to talk?”
“Of course,” Aarav said, laughing. “Can't I call my own father without a reason?”
His father laughed too. They talked about dinner, the weather, and a neighbor who had bought a new scooter. Nothing important happened during that conversation, yet Aarav felt lighter than he had all day.
Soon, the announcement for his train echoed through the station.
Aarav stood up and turned toward the old man.
“Thank you,” he said.
The old man looked confused. “For what?”
“For reminding me of something I should never have forgotten.”
The train arrived, its bright lights cutting through the rain. Aarav boarded and found a seat beside the window.
As the train moved, he watched the station disappear into the darkness. He wondered whether the old man would ever send that letter. He hoped he would.
That night, Aarav reached home later than usual, but he didn't open his laptop. Instead, he called his father again and asked him about his childhood.
For the first time in months, he listened without checking the time.
And as his father began telling an old story, Aarav realized something simple: life doesn't always give us a warning before moments become memories.
Sometimes, all we need is to stop running, sit beside a stranger, and remember what truly matters.
Because in the end, the most important journey isn't the one that takes us to work or toward success.
It's the journey that brings us closer to the people who are waiting for us at home.
"""
chunks = fixed_size_chunk(document, chunk_size=500, overlap=50)
print(f"Created {len(chunks)} chunks")
print(f"First chunk length: {len(chunks[0])} characters")


# Strategy 2: Recursive Chunking
# Instead of splitting only by character count, recursive chunking tries natural breakpoints first: paragraphs, then lines, then sentences, then words or characters. If a piece is still too large, it keeps splitting with smaller separators.

def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text recursively at natural boundaries."""
    separators = ["\n\n", "\n", ". ", " "]

    def split_text(text: str, separators: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        # Try each separator in order of preference
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current_chunk = ""

                for part in parts:
                    candidate = current_chunk + sep + part if current_chunk else part
                    if len(candidate) <= chunk_size:
                        current_chunk = candidate
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # If a single part exceeds chunk_size, split it further
                        if len(part) > chunk_size:
                            remaining_seps = separators[separators.index(sep) + 1:]
                            chunks.extend(split_text(part, remaining_seps))
                        else:
                            current_chunk = part

                if current_chunk:
                    chunks.append(current_chunk)
                return chunks

        # Fallback: character-level split
        return fixed_size_chunk(text, chunk_size, overlap)

    return split_text(text, separators)



chunks = recursive_chunk(document, chunk_size=120, overlap=20)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("-" * 40)
    
    
# Strategy 3: Semantic Chunking
# Semantic chunking uses a model to estimate where topics change instead of relying only on characters or formatting. One common approach is to embed consecutive sentences. When the similarity between adjacent sentences drops below a threshold, you treat that point as a possible topic boundary.

from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, threshold: float = 0.75) -> list[str]:
    """Split text at semantic boundaries using embedding similarity."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Split into sentences first
    sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if len(sentences) <= 1:
        return [text]

    # Embed all sentences
    embeddings = model.encode(sentences)

    # Compute cosine similarity between consecutive sentences
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        )

        if sim < threshold:
            # Similarity dropped: start a new chunk
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Add the last chunk
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

# Semantic chunking can produce coherent chunks when formatting is weak. It also adds cost: sentence embedding during ingestion, threshold tuning, more moving parts, and possible changes in behavior when the embedding model changes. Use it when document structure is poor enough to justify the extra complexity.


# Strategy 4: Document-Aware Chunking

# The most reliable strategy often uses the document's own structure: headers, sections, code blocks, lists, tables, page numbers, and captions. A Markdown file, HTML page, API reference, or technical manual already contains clues about topic boundaries. Use those clues when you can.


import re

def document_aware_chunk(markdown_text: str, max_chunk_size: int = 1000) -> list[dict]:
    """Split markdown by headers, preserving document structure."""
    # Split on headers (##, ###, etc.)
    sections = re.split(r'(^#{1,4}\s+.+$)', markdown_text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,4}\s+', part):
            # Save previous section
            if current_content.strip():
                chunks.append({
                    "header": current_header,
                    "content": current_content.strip(),
                    "metadata": {"section": current_header}
                })
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Save last section
    if current_content.strip():
        chunks.append({
            "header": current_header,
            "content": current_content.strip(),
            "metadata": {"section": current_header}
        })

    # Split sections that exceed max_chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > max_chunk_size:
            sub_chunks = recursive_chunk(chunk["content"], chunk_size=max_chunk_size)
            for i, sub in enumerate(sub_chunks):
                final_chunks.append({
                    "header": chunk["header"],
                    "content": sub,
                    "metadata": {**chunk["metadata"], "sub_part": i}
                })
        else:
            final_chunks.append(chunk)

    return final_chunks

markdown_text = """
# Introduction
Many modern applications use AI features such as recommendations, search,
classification, and conversational interfaces.

## What is AI Engineering?
AI engineering focuses on building reliable AI features for real users.
It combines machine learning, software engineering, and infrastructure design.

Engineers must handle data pipelines, model deployment, monitoring, and scaling.

## Vector Embeddings
Vector embeddings convert text into numerical representations.
They help systems compare the meaning of two pieces of text.

### Example Use Cases
Embeddings power search, recommendation systems, clustering, and semantic retrieval.

## Retrieval Augmented Generation (RAG)
RAG systems retrieve relevant documents from a knowledge base and provide them
as context to a language model to improve response accuracy.
"""

chunks = document_aware_chunk(markdown_text, max_chunk_size=200)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("Header:", chunk["header"])
    print("Metadata:", chunk["metadata"])
    print("Content:", chunk["content"])