"""
Unit tests for prompt_builder module.
No external dependencies — pure string construction logic.
Run with: pytest tests/unit/test_prompt_builder.py -v
"""

import pytest
from src.generation.prompt_builder import build_prompt, build_messages


# Reusable test fixtures
@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "Backpropagation computes gradients using the chain rule.",
            "source": "dl_textbook.pdf",
            "page": 4,
            "score": 0.82,
        },
        {
            "text": "The learning rate controls the size of weight updates.",
            "source": "dl_textbook.pdf",
            "page": 5,
            "score": 0.71,
        },
    ]


@pytest.fixture
def sample_query():
    return "How does backpropagation work?"


# ------------------------------------------------------------
# build_prompt tests
# ------------------------------------------------------------

def test_build_prompt_contains_query(sample_chunks, sample_query):
    """The query must appear in the prompt."""
    prompt = build_prompt(sample_query, sample_chunks)
    assert sample_query in prompt


def test_build_prompt_contains_chunk_text(sample_chunks, sample_query):
    """All chunk texts must appear in the prompt."""
    prompt = build_prompt(sample_query, sample_chunks)
    for chunk in sample_chunks:
        assert chunk["text"] in prompt


def test_build_prompt_contains_source_citations(sample_chunks, sample_query):
    """Source filename and page number must appear for each chunk."""
    prompt = build_prompt(sample_query, sample_chunks)
    for chunk in sample_chunks:
        assert chunk["source"] in prompt
        assert str(chunk["page"]) in prompt


def test_build_prompt_contains_system_instruction(sample_chunks, sample_query):
    """
    System prompt must contain grounding instruction.
    This is what prevents hallucination — must never be missing.
    """
    prompt = build_prompt(sample_query, sample_chunks)
    assert "ONLY" in prompt or "only" in prompt


def test_build_prompt_contains_fallback_instruction(sample_chunks, sample_query):
    """
    Fallback instruction must be present.
    Without it, LLM answers from training data when context is irrelevant.
    """
    prompt = build_prompt(sample_query, sample_chunks)
    assert "could not find" in prompt.lower() or "not in the" in prompt.lower()


def test_build_prompt_contains_separators(sample_chunks, sample_query):
    """Context and question separators must be present for clear structure."""
    prompt = build_prompt(sample_query, sample_chunks)
    assert "CONTEXT" in prompt
    assert "QUESTION" in prompt


def test_build_prompt_empty_chunks(sample_query):
    """Empty chunk list should still produce a valid prompt, not crash."""
    prompt = build_prompt(sample_query, [])
    assert sample_query in prompt
    assert len(prompt) > 0


def test_build_prompt_returns_string(sample_chunks, sample_query):
    """build_prompt must always return a string."""
    prompt = build_prompt(sample_query, sample_chunks)
    assert isinstance(prompt, str)


def test_build_prompt_chunk_ordering(sample_query):
    """Chunks should appear in order — Source 1 before Source 2."""
    chunks = [
        {"text": "First chunk text.", "source": "doc.pdf", "page": 1, "score": 0.9},
        {"text": "Second chunk text.", "source": "doc.pdf", "page": 2, "score": 0.8},
    ]
    prompt = build_prompt(sample_query, chunks)
    pos_first = prompt.index("First chunk text.")
    pos_second = prompt.index("Second chunk text.")
    assert pos_first < pos_second, "Chunks should appear in order"


# ------------------------------------------------------------
# build_messages tests
# ------------------------------------------------------------

def test_build_messages_returns_list(sample_chunks, sample_query):
    """build_messages must return a list."""
    messages = build_messages(sample_query, sample_chunks)
    assert isinstance(messages, list)


def test_build_messages_has_system_and_user(sample_chunks, sample_query):
    """Must have at least a system and user message."""
    messages = build_messages(sample_query, sample_chunks)
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_messages_user_contains_query(sample_chunks, sample_query):
    """User message must contain the query."""
    messages = build_messages(sample_query, sample_chunks)
    user_message = next(m for m in messages if m["role"] == "user")
    assert sample_query in user_message["content"]


def test_build_messages_each_message_has_role_and_content(sample_chunks, sample_query):
    """Every message must have both role and content fields."""
    messages = build_messages(sample_query, sample_chunks)
    for message in messages:
        assert "role" in message
        assert "content" in message
        assert message["content"]  # not empty