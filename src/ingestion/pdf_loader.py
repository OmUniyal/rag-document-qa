"""
Phase 1 of the pipeline: load raw PDFs and extract text.

Why this is its own module:
- PDF parsing is messy (headers, footers, tables, multi-column layouts).
- Separating loading from chunking means you can swap parsers without
  touching downstream code.
"""

from pathlib import Path
import fitz  # PyMuPDF
from src.utils.logger import logger


def load_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of dicts: [{"page": int, "text": str, "source": str}, ...]

    Why page-by-page?
        Preserves metadata (page number) for source attribution in answers.
        An answer citing "page 4 of document X" is far more useful than
        one with no provenance.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")
    pages = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                logger.debug(f"  Page {page_num}: empty (image-only or scanned?), skipping")
                continue
            pages.append({
                "page": page_num,
                "text": text,
                "source": path.name,
            })

    logger.info(f"  Extracted {len(pages)} pages from {path.name}")
    return pages


def load_pdfs_from_dir(dir_path: str | Path) -> list[dict]:
    """Load all PDFs from a directory. Used during bulk ingestion."""
    dir_path = Path(dir_path)
    all_pages = []
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {dir_path}")
        return []

    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)
        all_pages.extend(pages)

    logger.info(f"Total pages loaded: {len(all_pages)} from {len(pdf_files)} file(s)")
    return all_pages
