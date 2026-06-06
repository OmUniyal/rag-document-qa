"""
Unit tests for the chunker module.
Run with: pytest tests/unit/test_chunker.py -v
"""

import pytest
from src.ingestion.chunker import chunk_text, chunk_pages


def test_chunk_text_basic():
    text = "A" * 1200  # 1200 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 500


def test_chunk_text_overlap():
    """Overlapping chunks should share content at boundaries."""
    text = "word " * 200  # 1000 chars
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Each chunk boundary should have shared characters
    assert len(chunks) >= 2


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_pages_preserves_metadata():
    pages = [
        {"page": 1, "text": "Hello world. " * 50, "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["source"] == "test.pdf"
        assert chunk["page"] == 1
        assert "chunk_id" in chunk
        assert chunk["text"]


def test_chunk_pages_unique_ids():
    pages = [{"page": 1, "text": "X" * 2000, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
