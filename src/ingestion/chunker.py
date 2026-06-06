"""
Phase 2: split page text into overlapping chunks.

Key design questions this module answers (study these for interviews):
- Why chunk at all? LLMs have token limits; we can't feed a 100-page PDF.
- Why overlap? A sentence split across chunk boundaries would be missed
  by retrieval. Overlap ensures boundary context appears in at least one chunk.
- Why not just use token count? Characters are simpler; tokens vary by model.
  In production you'd switch to tiktoken-based splitting for precision.
"""

from src.utils.config import config
from src.utils.logger import logger


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Split a single string into overlapping chunks by character count.

    Args:
        text:       Raw text from one PDF page.
        chunk_size: Max characters per chunk. Defaults to config value.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        List of chunk strings.

    Interview note: This is a naive fixed-size chunker.
    A smarter approach splits at sentence boundaries (semantic chunking).
    We build this first to understand WHY the smarter approach matters.
    """
    chunk_size = chunk_size or config.chunk_size
    overlap = overlap or config.chunk_overlap

    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move start forward by (chunk_size - overlap) to create the overlap window
        start += chunk_size - overlap

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunk all pages from a document, preserving metadata.

    Returns:
        List of chunk dicts:
        [{"chunk_id": str, "text": str, "source": str, "page": int}, ...]

    The chunk_id is critical for deduplication and tracing answers back to source.
    """
    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = chunk_text(page["text"])
        for chunk in page_chunks:
            all_chunks.append({
                "chunk_id": f"{page['source']}_p{page['page']}_c{chunk_index}",
                "text": chunk,
                "source": page["source"],
                "page": page["page"],
            })
            chunk_index += 1

    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks
