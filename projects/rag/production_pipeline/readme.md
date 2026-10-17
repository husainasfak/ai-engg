The ingestion pipeline runs offline, on a schedule, or when documents change. It parses raw sources, normalizes text, splits content into chunks with metadata, embeds each chunk, and writes the result to the retrieval index.

The query pipeline runs on every user request. It takes a question, optionally preprocesses it, retrieves relevant chunks from the vector store, assembles them into a prompt with the question, and generates an answer


### Chunk metadata design

| Field | Type | Description | Example |
|---|---|---|---|
| source | string | Original file path | docs/api-reference.pdf |
| page | int | Page number (PDFs) | 12 |
| section | string | Section or heading title | Authentication |
| format | string | Source document format | pdf, markdown, html |
| chunk_index | int | Position within the document | 3 |
| chunk_total | int | Total chunks from this document | 47 |
| created_at | string | When the chunk was indexed | 2025-03-15T10:30:00Z |
| doc_title | string | Document title | API Reference v2.1 |
| char_start | int | Character offset in source | 4521 |




### For citations: source, page, and section let you point users to where an answer came from.

### For filtering: format and doc_title let you narrow searches to specific document types or individual documents. A user asking about the API can be routed to API docs.

### For debugging: chunk_index and chunk_total help you understand how documents were split and find adjacent chunks that might contain related information.

### For incremental updates: source and created_at help identify which chunks need to be re-indexed when a source document changes. Instead of re-indexing everything, you delete chunks matching the updated source and re-ingest that document.