"""
CLI script: ingest a PDF (or all PDFs in data/raw/) into the vector store.

Usage:
    python ingest.py                        # Ingest all PDFs in data/raw/
    python ingest.py --file data/raw/my.pdf # Ingest a single file
    python ingest.py --reset                # Clear vector store first, then ingest
"""

import argparse
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir
from src.ingestion.chunker import chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.utils.logger import logger


def ingest(file_path: str = None, reset: bool = False):
    store = VectorStore()

    if reset:
        logger.warning("Resetting vector store before ingestion.")
        store.reset()

    # Load pages
    if file_path:
        pages = load_pdf(file_path)
    else:
        pages = load_pdfs_from_dir("data/raw")

    if not pages:
        logger.error("No pages loaded. Check your PDF path.")
        return

    # Chunk
    chunks = chunk_pages(pages)
    logger.info(f"Total chunks to embed: {len(chunks)}")

    # Embed
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # Store
    store.add_chunks(chunks, embeddings)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store")
    parser.add_argument("--file", type=str, help="Path to a single PDF file")
    parser.add_argument("--reset", action="store_true", help="Reset vector store before ingesting")
    args = parser.parse_args()
    ingest(file_path=args.file, reset=args.reset)
