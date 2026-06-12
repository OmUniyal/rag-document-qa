"""
Unit tests for the evaluation module.
Run with: pytest tests/unit/test_ragas_eval.py -v
"""

import pytest
from src.evaluation.ragas_eval import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate,
)

# Shared test data
QUESTION = "How does backpropagation work?"
FAITHFUL_ANSWER = "Backpropagation computes gradients using the chain rule. The learning rate controls weight updates."
HALLUCINATED_ANSWER = "Backpropagation uses magic. The answer is always 42. Adam is always the best optimiser."
IRRELEVANT_ANSWER = "Paris is a beautiful city in France."

RELEVANT_CHUNKS = [
    "Backpropagation computes gradients by applying the chain rule recursively.",
    "The learning rate controls how large each weight update step is.",
]
IRRELEVANT_CHUNKS = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
]


# ------------------------------------------------------------
# faithfulness tests
# ------------------------------------------------------------

def test_faithfulness_returns_dict():
    result = faithfulness(FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert isinstance(result, dict)
    assert "score" in result
    assert "supported" in result
    assert "total" in result
    assert "details" in result


def test_faithfulness_score_range():
    result = faithfulness(FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert 0.0 <= result["score"] <= 1.0


def test_faithfulness_high_for_grounded_answer():
    result = faithfulness(FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert result["score"] >= 0.8, f"Expected high faithfulness, got {result['score']}"


def test_faithfulness_low_for_hallucinated_answer():
    result = faithfulness(HALLUCINATED_ANSWER, RELEVANT_CHUNKS)
    assert result["score"] < 0.8, f"Expected low faithfulness, got {result['score']}"


def test_faithfulness_empty_answer():
    result = faithfulness("", RELEVANT_CHUNKS)
    assert result["score"] == 0.0
    assert result["total"] == 0


# ------------------------------------------------------------
# answer_relevancy tests
# ------------------------------------------------------------

def test_answer_relevancy_returns_dict():
    result = answer_relevancy(QUESTION, FAITHFUL_ANSWER)
    assert isinstance(result, dict)
    assert "score" in result


def test_answer_relevancy_score_range():
    result = answer_relevancy(QUESTION, FAITHFUL_ANSWER)
    assert 0.0 <= result["score"] <= 1.0


def test_answer_relevancy_relevant_higher_than_irrelevant():
    relevant_score = answer_relevancy(QUESTION, FAITHFUL_ANSWER)["score"]
    irrelevant_score = answer_relevancy(QUESTION, IRRELEVANT_ANSWER)["score"]
    assert relevant_score > irrelevant_score, (
        f"Relevant answer ({relevant_score}) should score higher than irrelevant ({irrelevant_score})"
    )


# ------------------------------------------------------------
# context_precision tests
# ------------------------------------------------------------

def test_context_precision_returns_dict():
    result = context_precision(QUESTION, RELEVANT_CHUNKS)
    assert isinstance(result, dict)
    assert "score" in result
    assert "relevant" in result
    assert "total" in result


def test_context_precision_score_range():
    result = context_precision(QUESTION, RELEVANT_CHUNKS)
    assert 0.0 <= result["score"] <= 1.0


def test_context_precision_relevant_chunks_score_higher():
    relevant_score = context_precision(QUESTION, RELEVANT_CHUNKS)["score"]
    irrelevant_score = context_precision(QUESTION, IRRELEVANT_CHUNKS)["score"]
    assert relevant_score > irrelevant_score, (
        f"Relevant chunks ({relevant_score}) should score higher than irrelevant ({irrelevant_score})"
    )


def test_context_precision_empty_chunks():
    result = context_precision(QUESTION, [])
    assert result["score"] == 0.0


# ------------------------------------------------------------
# context_recall tests
# ------------------------------------------------------------

def test_context_recall_returns_dict():
    result = context_recall(FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert isinstance(result, dict)
    assert "score" in result


def test_context_recall_score_range():
    result = context_recall(FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert 0.0 <= result["score"] <= 1.0


def test_context_recall_high_for_supported_answer():
    result = context_recall(FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert result["score"] >= 0.7, f"Expected high recall, got {result['score']}"


def test_context_recall_low_for_unsupported_answer():
    result = context_recall(FAITHFUL_ANSWER, IRRELEVANT_CHUNKS)
    assert result["score"] < 0.5, f"Expected low recall, got {result['score']}"


# ------------------------------------------------------------
# evaluate (combined) tests
# ------------------------------------------------------------

def test_evaluate_returns_all_metrics():
    result = evaluate(QUESTION, FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert "overall" in result
    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert "context_precision" in result
    assert "context_recall" in result
    assert "details" in result


def test_evaluate_overall_score_range():
    result = evaluate(QUESTION, FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    assert 0.0 <= result["overall"] <= 1.0


def test_evaluate_faithful_answer_scores_higher_than_hallucinated():
    faithful_result = evaluate(QUESTION, FAITHFUL_ANSWER, RELEVANT_CHUNKS)
    hallucinated_result = evaluate(QUESTION, HALLUCINATED_ANSWER, RELEVANT_CHUNKS)
    assert faithful_result["overall"] > hallucinated_result["overall"], (
        f"Faithful ({faithful_result['overall']}) should score higher than hallucinated ({hallucinated_result['overall']})"
    )