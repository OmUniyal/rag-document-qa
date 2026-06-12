"""
Unit tests for pdf_loader module.
Creates a real temporary PDF for testing — no external file dependency.
Run with: pytest tests/unit/test_pdf_loader.py -v
"""

import pytest
from pathlib import Path
import tempfile
import os
from src.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir


# ------------------------------------------------------------
# Helpers — create minimal test PDFs in a temp directory
# ------------------------------------------------------------

def create_test_pdf(path: Path, content: str = "This is a test document."):
    """
    Create a minimal valid PDF file at the given path.
    Uses reportlab if available, falls back to fpdf2.
    """
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, content)
        pdf.output(str(path))
    except ImportError:
        pytest.skip("fpdf2 not installed — skipping PDF loader tests")


@pytest.fixture(scope="module")
def temp_pdf_dir():
    """Create a temporary directory with test PDFs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Single page PDF
        create_test_pdf(
            tmpdir_path / "single_page.pdf",
            "Neural networks learn by adjusting weights."
        )

        # Multi-page PDF
        try:
            from fpdf import FPDF
            pdf = FPDF()
            for i in range(3):
                pdf.add_page()
                pdf.set_font("Helvetica", size=12)
                pdf.cell(0, 10, f"Page {i+1} content about machine learning.")
            pdf.output(str(tmpdir_path / "multi_page.pdf"))
        except ImportError:
            pytest.skip("fpdf2 not installed")

        yield tmpdir_path


@pytest.fixture(scope="module")
def single_pdf_path(temp_pdf_dir):
    return temp_pdf_dir / "single_page.pdf"


@pytest.fixture(scope="module")
def multi_pdf_path(temp_pdf_dir):
    return temp_pdf_dir / "multi_page.pdf"


# ------------------------------------------------------------
# load_pdf tests
# ------------------------------------------------------------

def test_load_pdf_returns_list(single_pdf_path):
    """load_pdf must return a list."""
    pages = load_pdf(single_pdf_path)
    assert isinstance(pages, list)


def test_load_pdf_returns_pages_with_required_fields(single_pdf_path):
    """Each page must have text, page number, and source."""
    pages = load_pdf(single_pdf_path)
    assert len(pages) > 0
    for page in pages:
        assert "text" in page
        assert "page" in page
        assert "source" in page


def test_load_pdf_page_numbers_start_at_1(single_pdf_path):
    """Page numbers should be 1-indexed, not 0-indexed."""
    pages = load_pdf(single_pdf_path)
    assert pages[0]["page"] == 1


def test_load_pdf_source_is_filename(single_pdf_path):
    """Source should be the filename, not the full path."""
    pages = load_pdf(single_pdf_path)
    assert pages[0]["source"] == "single_page.pdf"
    assert "\\" not in pages[0]["source"]
    assert "/" not in pages[0]["source"]


def test_load_pdf_text_is_non_empty(single_pdf_path):
    """Extracted text should not be empty."""
    pages = load_pdf(single_pdf_path)
    for page in pages:
        assert page["text"].strip()


def test_load_pdf_multipage_returns_all_pages(multi_pdf_path):
    """Multi-page PDF should return one dict per page."""
    pages = load_pdf(multi_pdf_path)
    assert len(pages) == 3


def test_load_pdf_multipage_page_numbers_sequential(multi_pdf_path):
    """Page numbers should be sequential: 1, 2, 3..."""
    pages = load_pdf(multi_pdf_path)
    page_numbers = [p["page"] for p in pages]
    assert page_numbers == list(range(1, len(pages) + 1))


def test_load_pdf_file_not_found():
    """Should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent_file.pdf")


def test_load_pdf_wrong_extension(tmp_path):
    """Should raise ValueError for non-PDF files."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("some content")
    with pytest.raises(ValueError):
        load_pdf(txt_file)


# ------------------------------------------------------------
# load_pdfs_from_dir tests
# ------------------------------------------------------------

def test_load_pdfs_from_dir_returns_list(temp_pdf_dir):
    """Should return a flat list of all pages from all PDFs."""
    pages = load_pdfs_from_dir(temp_pdf_dir)
    assert isinstance(pages, list)
    assert len(pages) > 0


def test_load_pdfs_from_dir_empty_directory(tmp_path):
    """Empty directory should return empty list, not crash."""
    pages = load_pdfs_from_dir(tmp_path)
    assert pages == []


def test_load_pdfs_from_dir_combines_all_pdfs(temp_pdf_dir):
    """Should load pages from all PDFs in directory."""
    pages = load_pdfs_from_dir(temp_pdf_dir)
    sources = {p["source"] for p in pages}
    assert "single_page.pdf" in sources
    assert "multi_page.pdf" in sources