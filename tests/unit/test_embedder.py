"""
Unit tests for the embedder module.
These tests run without needing Ollama or a GPU.
"""

import pytest
from src.retrieval.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()  # Downloads model on first run, cached after


def test_embed_query_returns_vector(embedder):
    vec = embedder.embed_query("What is RAG?")
    assert isinstance(vec, list)
    assert len(vec) == embedder.embedding_dim
    assert all(isinstance(v, float) for v in vec)


def test_embed_texts_batch(embedder):
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    vecs = embedder.embed_texts(texts)
    assert len(vecs) == 3
    assert all(len(v) == embedder.embedding_dim for v in vecs)


def test_embed_texts_empty(embedder):
    assert embedder.embed_texts([]) == []


def test_similar_sentences_closer_than_dissimilar(embedder):
    """Semantic sanity check: related texts should have higher cosine similarity."""
    import numpy as np

    v1 = np.array(embedder.embed_query("The cat sat on the mat"))
    v2 = np.array(embedder.embed_query("A feline rested on the rug"))
    v3 = np.array(embedder.embed_query("Quantum entanglement in physics"))

    cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sim_related = cos(v1, v2)
    sim_unrelated = cos(v1, v3)

    assert sim_related > sim_unrelated, (
        f"Expected related pair ({sim_related:.3f}) > unrelated pair ({sim_unrelated:.3f})"
    )
