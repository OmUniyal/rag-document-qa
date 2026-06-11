"""
PDF loader using pdfplumber — pure Python, no native DLLs.
Extracts text page by page, preserving page number metadata
for source attribution in answers.
"""

from pathlib import Path
import pdfplumber
from src.utils.logger import logger


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of dicts: [{"page": int, "text": str, "source": str}, ...]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")
    pages = []

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                logger.debug(f"  Page {page_num}: empty, skipping")
                continue
            pages.append({
                "page": page_num,
                "text": text.strip(),
                "source": path.name,
            })

    logger.info(f"  Extracted {len(pages)} pages from {path.name}")
    return pages


def load_pdfs_from_dir(dir_path: str) -> list[dict]:
    """Load all PDFs from a directory."""
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