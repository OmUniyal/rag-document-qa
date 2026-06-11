"""
Integration test — runs the full RAG pipeline end to end.
Requires: ChromaDB populated (run ingest.py first), GROQ_API_KEY in .env

Run with: pytest tests/integration/ -v
"""

import pytest
from src.generation.rag_chain import RAGChain
from src.retrieval.retriever import Retriever
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdfs_from_dir


# ------------------------------------------------------------
# Retriever integration tests — no LLM call needed
# ------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    """Shared retriever instance for all tests in this module."""
    return Retriever()


def test_retriever_returns_results(retriever):
    """Basic sanity check — retriever returns chunks for a valid query."""
    results = retriever.retrieve("What is YOLO?", top_k=3)
    assert len(results) > 0, "Retriever returned no results"


def test_retriever_returns_correct_fields(retriever):
    """Each result must have the fields the rest of the pipeline depends on."""
    results = retriever.retrieve("What is YOLO?", top_k=3)
    for r in results:
        assert "text" in r, "Missing 'text' field"
        assert "source" in r, "Missing 'source' field"
        assert "page" in r, "Missing 'page' field"
        assert "score" in r, "Missing 'score' field"


def test_retriever_scores_between_0_and_1(retriever):
    """Cosine similarity scores should be in [0, 1] range."""
    results = retriever.retrieve("object detection", top_k=5)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"


def test_retriever_top_k_respected(retriever):
    """Retriever should return at most top_k results."""
    results = retriever.retrieve("neural networks", top_k=3)
    assert len(results) <= 3


def test_retriever_relevant_content_for_known_query(retriever):
    """
    For a query we know is in the CV study guide,
    at least one result should have a high similarity score.
    """
    results = retriever.retrieve("YOLO single stage detector", top_k=3)
    top_score = results[0]["score"]
    assert top_score > 0.5, (
        f"Expected top score > 0.5 for known query, got {top_score}"
    )


def test_retriever_empty_query_handled(retriever):
    """
    Empty query should not crash — may return results or empty list.
    The important thing is no unhandled exception.
    """
    try:
        results = retriever.retrieve("", top_k=3)
        # If it returns results, that's acceptable behaviour
        # (vector store returns nearest neighbours to zero vector)
    except Exception as e:
        pytest.fail(f"Empty query raised unexpected exception: {e}")


# ------------------------------------------------------------
# RAG chain integration tests — makes real LLM calls
# Mark with 'llm' so they can be skipped in CI
# pytest tests/integration/ -v -m "not llm"
# ------------------------------------------------------------

@pytest.fixture(scope="module")
def rag_chain():
    return RAGChain()


@pytest.mark.llm
def test_rag_chain_returns_answer(rag_chain):
    """Full pipeline returns a non-empty answer."""
    result = rag_chain.query("What is YOLO?")
    assert result["answer"], "Answer should not be empty"
    assert len(result["answer"]) > 20, "Answer seems too short"


@pytest.mark.llm
def test_rag_chain_returns_sources(rag_chain):
    """Answer should include source attribution."""
    result = rag_chain.query("What is YOLO?")
    assert len(result["sources"]) > 0, "No sources returned"
    for s in result["sources"]:
        assert "source" in s
        assert "page" in s
        assert "score" in s


@pytest.mark.llm
def test_rag_chain_grounding(rag_chain):
    """
    Query about something not in the document should trigger
    the fallback response, not a hallucinated answer.
    """
    result = rag_chain.query(
        "What is the capital of Australia?"
    )
    answer = result["answer"].lower()
    # Should say it couldn't find the answer, not confidently say "Canberra"
    grounded_phrases = [
        "could not find",
        "not in the",
        "not mentioned",
        "no information",
        "cannot find",
    ]
    is_grounded = any(phrase in answer for phrase in grounded_phrases)
    assert is_grounded, (
        f"Expected grounded fallback response, got: {result['answer']}"
    )


@pytest.mark.llm
def test_rag_chain_result_structure(rag_chain):
    """Result dict should always have required keys."""
    result = rag_chain.query("What is U-Net?")
    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert "chunks_used" in result
    assert result["chunks_used"] > 0