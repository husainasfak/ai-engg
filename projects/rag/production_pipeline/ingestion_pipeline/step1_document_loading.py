import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Document:
    """A loaded document with text content and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return one Document per page."""
    try:
        import pymupdf  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    docs = []
    pdf = pymupdf.open(file_path)
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()
        if text.strip():  # Skip empty pages
            docs.append(Document(
                content=text,
                metadata={
                    "source": file_path,
                    "page": page_num + 1,
                    "total_pages": len(pdf),
                    "format": "pdf",
                }
            ))
    pdf.close()
    return docs

def load_markdown(file_path: str) -> list[Document]:
    """Load a Markdown file and return one Document per section."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by top-level headers (## or #)
    sections = []
    current_section = ""
    current_title = "Introduction"

    for line in content.split("\n"):
        if line.startswith("# ") or line.startswith("## "):
            if current_section.strip():
                sections.append((current_title, current_section.strip()))
            current_title = line.lstrip("# ").strip()
            current_section = line + "\n"
        else:
            current_section += line + "\n"

    if current_section.strip():
        sections.append((current_title, current_section.strip()))

    return [
        Document(
            content=text,
            metadata={
                "source": file_path,
                "section": title,
                "section_index": i,
                "format": "markdown",
            }
        )
        for i, (title, text) in enumerate(sections)
    ]

def load_html(file_path: str) -> list[Document]:
    """Load an HTML file, strip tags, return as a single Document."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install BeautifulSoup: pip install beautifulsoup4")

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Remove script, style, and nav elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return [Document(
        content=text,
        metadata={"source": file_path, "format": "html"}
    )]

def load_document(file_path: str) -> list[Document]:
    """Load a document based on its file extension."""
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".md": load_markdown,
        ".html": load_html,
        ".htm": load_html,
    }

    loader = loaders.get(ext)
    if loader is None:
        # Fall back to plain text
        with open(file_path, "r", encoding="utf-8") as f:
            return [Document(
                content=f.read(),
                metadata={"source": file_path, "format": "text"}
            )]
    return loader(file_path)