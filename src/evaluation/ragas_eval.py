"""
RAG Evaluation — lightweight implementation of RAGAS-style metrics.

Why not use the ragas library directly:
  ragas has unstable langchain_community dependencies that break frequently.
  This module implements the same four core metrics from first principles
  using only sentence-transformers and the Groq LLM we already have.

Metrics implemented:
  1. Faithfulness      — are answer claims supported by context?
  2. Answer Relevancy  — does the answer address the question?
  3. Context Precision — are retrieved chunks relevant to the question?
  4. Context Recall    — were all needed chunks retrieved?
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger

# Use the same embedding model as the retrieval pipeline
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------
# Metric 1: Faithfulness
# "Is every claim in the answer supported by the retrieved context?"
# Score 0-1. Low score = hallucination.
# ------------------------------------------------------------

def faithfulness(answer: str, context_chunks: list[str], threshold: float = 0.5) -> dict:
    """
    Checks if each sentence in the answer has semantic support
    in at least one retrieved chunk.

    Args:
        answer:         The LLM-generated answer string.
        context_chunks: List of retrieved chunk texts.
        threshold:      Minimum cosine similarity to consider a sentence supported.

    Returns:
        {
            "score": float,           # 0-1, fraction of sentences supported
            "supported": int,         # number of supported sentences
            "total": int,             # total sentences checked
            "details": list[dict]     # per-sentence breakdown
        }

    Note: Production RAGAS uses an LLM to decompose answers into atomic claims.
    This implementation uses embedding similarity as a faster approximation.
    """
    model = _get_embedder()

    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return {"score": 0.0, "supported": 0, "total": 0, "details": []}

    chunk_embeddings = model.encode(context_chunks)
    details = []
    supported = 0

    for sentence in sentences:
        sentence_embedding = model.encode(sentence)
        max_similarity = max(
            _cosine_similarity(sentence_embedding, chunk_emb)
            for chunk_emb in chunk_embeddings
        )
        is_supported = max_similarity >= threshold
        if is_supported:
            supported += 1
        details.append({
            "sentence": sentence,
            "max_similarity": round(max_similarity, 4),
            "supported": is_supported,
        })

    score = supported / len(sentences)
    logger.debug(f"Faithfulness: {score:.2f} ({supported}/{len(sentences)} sentences supported)")
    return {
        "score": round(score, 4),
        "supported": supported,
        "total": len(sentences),
        "details": details,
    }


# ------------------------------------------------------------
# Metric 2: Answer Relevancy
# "Does the answer actually address the question asked?"
# Score 0-1. Low score = answer is off-topic or evasive.
# ------------------------------------------------------------

def answer_relevancy(question: str, answer: str) -> dict:
    """
    Measures semantic similarity between the question and answer.
    High similarity = answer directly addresses the question.

    Note: Production RAGAS generates multiple questions from the answer
    and measures how well they reconstruct the original question.
    This is a simpler direct similarity approximation.
    """
    model = _get_embedder()

    question_embedding = model.encode(question)
    answer_embedding = model.encode(answer)
    score = _cosine_similarity(question_embedding, answer_embedding)

    # Clamp to [0, 1] — cosine can be slightly negative
    score = max(0.0, min(1.0, score))

    logger.debug(f"Answer relevancy: {score:.2f}")
    return {
        "score": round(score, 4),
        "question": question,
        "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
    }


# ------------------------------------------------------------
# Metric 3: Context Precision
# "Are the retrieved chunks actually relevant to the question?"
# Score 0-1. Low score = too many irrelevant chunks retrieved.
# ------------------------------------------------------------

def context_precision(question: str, context_chunks: list[str], threshold: float = 0.4) -> dict:
    """
    Measures what fraction of retrieved chunks are relevant to the question.
    Signal-to-noise ratio of retrieval.

    Args:
        question:       The user's question.
        context_chunks: Retrieved chunk texts.
        threshold:      Minimum similarity to consider a chunk relevant.
    """
    if not context_chunks:
        return {"score": 0.0, "relevant": 0, "total": 0, "details": []}

    model = _get_embedder()
    question_embedding = model.encode(question)

    details = []
    relevant = 0

    for i, chunk in enumerate(context_chunks):
        chunk_embedding = model.encode(chunk)
        similarity = _cosine_similarity(question_embedding, chunk_embedding)
        is_relevant = similarity >= threshold
        if is_relevant:
            relevant += 1
        details.append({
            "chunk_index": i,
            "similarity": round(similarity, 4),
            "relevant": is_relevant,
            "preview": chunk[:80] + "..." if len(chunk) > 80 else chunk,
        })

    score = relevant / len(context_chunks)
    logger.debug(f"Context precision: {score:.2f} ({relevant}/{len(context_chunks)} chunks relevant)")
    return {
        "score": round(score, 4),
        "relevant": relevant,
        "total": len(context_chunks),
        "details": details,
    }


# ------------------------------------------------------------
# Metric 4: Context Recall
# "Did we retrieve all the chunks needed to answer fully?"
# Score 0-1. Low score = answer is incomplete due to missing context.
# ------------------------------------------------------------

def context_recall(answer: str, context_chunks: list[str], threshold: float = 0.5) -> dict:
    """
    Measures what fraction of the answer's content is attributable
    to the retrieved context. Inverse of faithfulness framing —
    here we measure coverage rather than support.

    Note: Production RAGAS uses ground truth answers for recall.
    Without ground truth, we approximate by checking how much of
    the answer is semantically covered by the context.
    """
    model = _get_embedder()

    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences or not context_chunks:
        return {"score": 0.0, "covered": 0, "total": 0}

    chunk_embeddings = model.encode(context_chunks)
    covered = 0

    for sentence in sentences:
        sentence_embedding = model.encode(sentence)
        max_similarity = max(
            _cosine_similarity(sentence_embedding, chunk_emb)
            for chunk_emb in chunk_embeddings
        )
        if max_similarity >= threshold:
            covered += 1

    score = covered / len(sentences)
    logger.debug(f"Context recall: {score:.2f} ({covered}/{len(sentences)} sentences covered)")
    return {
        "score": round(score, 4),
        "covered": covered,
        "total": len(sentences),
    }


# ------------------------------------------------------------
# Combined evaluation — run all four metrics at once
# ------------------------------------------------------------

def evaluate(
    question: str,
    answer: str,
    context_chunks: list[str],
    faithfulness_threshold: float = 0.5,
    precision_threshold: float = 0.4,
) -> dict:
    """
    Run all four metrics and return a combined evaluation report.

    Args:
        question:        The user's question.
        answer:          The LLM-generated answer.
        context_chunks:  List of retrieved chunk texts.

    Returns:
        Dict with all four metric results and an overall summary.
    """
    logger.info("Running RAG evaluation...")

    faith = faithfulness(answer, context_chunks, faithfulness_threshold)
    relevancy = answer_relevancy(question, answer)
    precision = context_precision(question, context_chunks, precision_threshold)
    recall = context_recall(answer, context_chunks, faithfulness_threshold)

    # Overall score — average of four metrics
    overall = np.mean([
        faith["score"],
        relevancy["score"],
        precision["score"],
        recall["score"],
    ])

    report = {
        "overall": round(float(overall), 4),
        "faithfulness": faith["score"],
        "answer_relevancy": relevancy["score"],
        "context_precision": precision["score"],
        "context_recall": recall["score"],
        "details": {
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
        }
    }

    logger.info(
        f"Evaluation complete — "
        f"overall={report['overall']:.2f} | "
        f"faithfulness={faith['score']:.2f} | "
        f"relevancy={relevancy['score']:.2f} | "
        f"precision={precision['score']:.2f} | "
        f"recall={recall['score']:.2f}"
    )

    return report