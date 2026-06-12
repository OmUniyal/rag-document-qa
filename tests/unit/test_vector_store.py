"""
Unit tests for vector_store module.
Uses an isolated in-memory-style collection that gets cleaned up after each test.
Run with: pytest tests/unit/test_vector_store.py -v
"""

import pytest
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def store(tmp_path):
    """
    Fresh vector store for each test — uses tmp_path so tests
    don't interfere with each other or the production chroma_db.
    """
    vs = VectorStore(
        collection_name="test_collection",
        persist_dir=str(tmp_path / "test_chroma")
    )
    return vs


@pytest.fixture
def populated_store(store, embedder):
    """Vector store with some chunks already added."""
    chunks = [
        {"chunk_id": "doc1_p1_c1", "text": "Neural networks learn by backpropagation.", "source": "test.pdf", "page": 1},
        {"chunk_id": "doc1_p1_c2", "text": "The learning rate controls weight update size.", "source": "test.pdf", "page": 2},
        {"chunk_id": "doc1_p1_c3", "text": "Gradient descent minimises the loss function.", "source": "test.pdf", "page": 3},
        {"chunk_id": "doc2_p1_c1", "text": "Paris is the capital of France.", "source": "geography.pdf", "page": 1},
        {"chunk_id": "doc2_p1_c2", "text": "The Eiffel Tower is located in Paris.", "source": "geography.pdf", "page": 2},
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(chunks, embeddings)
    return store


# ------------------------------------------------------------
# add_chunks tests
# ------------------------------------------------------------

def test_add_chunks_increases_count(store, embedder):
    """Adding chunks should increase collection count."""
    assert store.collection.count() == 0
    chunks = [
        {"chunk_id": "c1", "text": "Hello world.", "source": "test.pdf", "page": 1}
    ]
    embeddings = embedder.embed_texts(["Hello world."])
    store.add_chunks(chunks, embeddings)
    assert store.collection.count() == 1


def test_add_chunks_empty_list(store):
    """Adding empty list should not crash or change count."""
    store.add_chunks([], [])
    assert store.collection.count() == 0


def test_add_multiple_chunks(store, embedder):
    """Adding multiple chunks should store all of them."""
    chunks = [
        {"chunk_id": f"c{i}", "text": f"Chunk {i} text.", "source": "test.pdf", "page": i}
        for i in range(5)
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(chunks, embeddings)
    assert store.collection.count() == 5


# ------------------------------------------------------------
# query tests
# ------------------------------------------------------------

def test_query_returns_results(populated_store, embedder):
    """Query should return a non-empty list of results."""
    query_embedding = embedder.embed_query("How do neural networks learn?")
    results = populated_store.query(query_embedding, top_k=3)
    assert len(results) > 0


def test_query_returns_correct_fields(populated_store, embedder):
    """Each result must have text, source, page, and score fields."""
    query_embedding = embedder.embed_query("neural networks")
    results = populated_store.query(query_embedding, top_k=3)
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "page" in r
        assert "score" in r


def test_query_top_k_respected(populated_store, embedder):
    """Query should return at most top_k results."""
    query_embedding = embedder.embed_query("machine learning")
    results = populated_store.query(query_embedding, top_k=2)
    assert len(results) <= 2


def test_query_scores_in_valid_range(populated_store, embedder):
    """Scores should be between 0 and 1."""
    query_embedding = embedder.embed_query("neural networks")
    results = populated_store.query(query_embedding, top_k=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"


def test_query_results_ordered_by_score(populated_store, embedder):
    """Results should be ordered by score descending."""
    query_embedding = embedder.embed_query("neural networks backpropagation")
    results = populated_store.query(query_embedding, top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not ordered by score"


def test_query_relevant_content_ranks_higher(populated_store, embedder):
    """
    A query about neural networks should return ML chunks
    above geography chunks.
    """
    query_embedding = embedder.embed_query("How does gradient descent work?")
    results = populated_store.query(query_embedding, top_k=3)
    top_result = results[0]
    assert top_result["source"] == "test.pdf", (
        f"Expected ML content to rank first, got: {top_result['text']}"
    )


# ------------------------------------------------------------
# reset tests
# ------------------------------------------------------------

def test_reset_clears_collection(populated_store):
    """Reset should empty the collection."""
    assert populated_store.collection.count() > 0
    populated_store.reset()
    assert populated_store.collection.count() == 0