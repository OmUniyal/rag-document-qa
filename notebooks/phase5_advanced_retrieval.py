# =============================================================
# PHASE 5: Advanced Retrieval — fixing the two core weaknesses
# Run with: python notebooks\phase5_advanced_retrieval.py
# =============================================================

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# The same chunks from Phase 1 — our baseline
chunks = [
    "The transformer architecture uses self-attention mechanisms.",
    "BERT is a bidirectional transformer trained on masked language modelling.",
    "Gradient descent optimises neural network weights by minimising loss.",
    "Photosynthesis is the process by which plants convert sunlight to energy.",
    "The Eiffel Tower is located in Paris, France.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Backpropagation computes gradients by the chain rule.",
    "The Seine river flows through the city of Paris.",
]

chunk_embeddings = model.encode(chunks)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dense_retrieve(query, top_k=3):
    qe = model.encode(query)
    scores = [(cosine_sim(qe, ce), i, chunks[i]) for i, ce in enumerate(chunk_embeddings)]
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# ------------------------------------------------------------
# BLOCK 1: Reproduce the failure — vocabulary mismatch
# ------------------------------------------------------------
print("=" * 60)
print("BLOCK 1: The vocabulary mismatch failure")
print("=" * 60)

query = "How are model parameters updated during training?"
results = dense_retrieve(query, top_k=3)

print(f"\nQuery: '{query}'")
print("Dense retrieval results:")
for score, idx, chunk in results:
    marker = "✓" if "gradient" in chunk.lower() or "backprop" in chunk.lower() else "✗"
    print(f"  [{marker}] score={score:.4f} | {chunk}")

print("""
Problem: "model parameters updated" doesn't match "neural network weights minimising"
The correct answers (gradient descent, backpropagation) are buried below attention.
This is vocabulary mismatch — same concept, different words.
""")

# ------------------------------------------------------------
# BLOCK 2: BM25 sparse retrieval — keyword matching
# ------------------------------------------------------------
print("=" * 60)
print("BLOCK 2: BM25 — sparse retrieval fixes keyword matching")
print("=" * 60)

# pip install rank_bm25
from rank_bm25 import BM25Okapi

# BM25 works on token lists, not embeddings
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def sparse_retrieve(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], i, chunks[i]) for i in top_indices]

print(f"\nQuery: '{query}'")
print("BM25 sparse retrieval results:")
bm25_results = sparse_retrieve(query, top_k=3)
for score, idx, chunk in bm25_results:
    marker = "✓" if "gradient" in chunk.lower() or "backprop" in chunk.lower() else "✗"
    print(f"  [{marker}] score={score:.4f} | {chunk}")

print("""
BM25 rewards exact keyword matches.
"training" appears in chunks about training — scores those higher.
But BM25 misses semantic similarity — "feline" won't match "cat".
Neither dense nor sparse is complete alone.
""")

# ------------------------------------------------------------
# BLOCK 3: Hybrid search — combining both
# ------------------------------------------------------------
print("=" * 60)
print("BLOCK 3: Hybrid search — best of both worlds")
print("=" * 60)

def hybrid_retrieve(query, top_k=3, alpha=0.5):
    """
    Combine dense + sparse retrieval scores.
    alpha=0.5 weights them equally.
    alpha=0.7 favours dense (semantic).
    alpha=0.3 favours sparse (keyword).
    
    Scores are normalised to [0,1] before combining.
    """
    # Dense scores
    qe = model.encode(query)
    dense_scores = np.array([cosine_sim(qe, ce) for ce in chunk_embeddings])
    
    # Sparse scores
    tokenized_query = query.lower().split()
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Normalise both to [0, 1]
    def normalise(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return scores
        return (scores - min_s) / (max_s - min_s)
    
    dense_norm = normalise(dense_scores)
    sparse_norm = normalise(sparse_scores)
    
    # Weighted combination
    combined = alpha * dense_norm + (1 - alpha) * sparse_norm
    
    top_indices = np.argsort(combined)[::-1][:top_k]
    return [(combined[i], dense_scores[i], sparse_scores[i], chunks[i]) 
            for i in top_indices]

print(f"\nQuery: '{query}'")
print("Hybrid retrieval results (alpha=0.5):")
hybrid_results = hybrid_retrieve(query, top_k=3, alpha=0.5)
for combined, dense, sparse, chunk in hybrid_results:
    marker = "✓" if "gradient" in chunk.lower() or "backprop" in chunk.lower() else "✗"
    print(f"  [{marker}] combined={combined:.4f} | dense={dense:.4f} | sparse={sparse:.4f}")
    print(f"       {chunk}")

print("\nHybrid retrieval results (alpha=0.7 — favour semantic):")
hybrid_results_07 = hybrid_retrieve(query, top_k=3, alpha=0.7)
for combined, dense, sparse, chunk in hybrid_results_07:
    marker = "✓" if "gradient" in chunk.lower() or "backprop" in chunk.lower() else "✗"
    print(f"  [{marker}] combined={combined:.4f} | {chunk}")

# ------------------------------------------------------------
# BLOCK 4: Re-ranking with a cross-encoder
# Retrieve wide, then re-rank precisely
# ------------------------------------------------------------
print("=" * 60)
print("BLOCK 4: Re-ranking — retrieve wide, re-rank precisely")
print("=" * 60)

print("""
The two-stage approach:
Stage 1 — Retrieve: use dense retrieval to get top-10 candidates fast
Stage 2 — Re-rank: use a cross-encoder to re-score top-10 precisely

Why cross-encoders are more accurate:
- Bi-encoder (what we've been using): embeds query and chunk SEPARATELY,
  then compares vectors. Fast but loses interaction between query and chunk.
- Cross-encoder: reads query AND chunk TOGETHER in one forward pass.
  Sees the full interaction. Much more accurate but too slow for full corpus.

This is why we use bi-encoder for retrieval (speed) 
and cross-encoder for re-ranking (accuracy).
""")

from sentence_transformers import CrossEncoder

# This model is trained specifically to score query-chunk relevance
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidate_chunks, top_k=3):
    """
    Score each candidate chunk against the query using a cross-encoder.
    Returns top_k most relevant chunks.
    """
    pairs = [[query, chunk] for chunk in candidate_chunks]
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(
        zip(scores, candidate_chunks),
        key=lambda x: x[0],
        reverse=True
    )
    return ranked[:top_k]


# Stage 1: retrieve wider candidate set — top 6 instead of top 3
print(f"Query: '{query}'")
print("\nStage 1 — Dense retrieval, top-6 candidates:")
wide_results = dense_retrieve(query, top_k=6)
candidate_chunks = [chunk for _, _, chunk in wide_results]
for score, idx, chunk in wide_results:
    marker = "✓" if "gradient" in chunk.lower() or "backprop" in chunk.lower() else "✗"
    print(f"  [{marker}] score={score:.4f} | {chunk}")

# Stage 2: re-rank with cross-encoder
print("\nStage 2 — After cross-encoder re-ranking, top-3:")
reranked = rerank(query, candidate_chunks, top_k=3)
for score, chunk in reranked:
    marker = "✓" if "gradient" in chunk.lower() or "backprop" in chunk.lower() else "✗"
    print(f"  [{marker}] score={score:.4f} | {chunk}")

print("""
Cross-encoder reads "How are model parameters updated during training?"
alongside "Gradient descent optimises neural network weights by minimising loss."
and understands they're about the same concept — even with different words.
This is the key difference from bi-encoder similarity.
""")

# ------------------------------------------------------------
# BLOCK 5: Evaluation — how do you measure RAG quality?
# RAGAS metrics: faithfulness, answer relevancy,
#                context precision, context recall
# ------------------------------------------------------------
print("=" * 60)
print("BLOCK 5: RAG Evaluation — measuring what actually matters")
print("=" * 60)

print("""
The 4 RAGAS metrics — what each measures:

1. FAITHFULNESS
   "Is the answer supported by the retrieved context?"
   Checks if every claim in the answer can be traced back to a chunk.
   Score 0-1. Low score = hallucination.
   
   Example:
   Context: "Paris is the capital of France"
   Answer:  "Paris is the capital of France and has 3M people"
   → Low faithfulness: "3M people" not in context

2. ANSWER RELEVANCY  
   "Does the answer actually address the question asked?"
   Score 0-1. Low score = answer is off-topic or evasive.
   
   Example:
   Question: "What is backpropagation?"
   Answer:   "Neural networks are very useful in practice."
   → Low relevancy: doesn't answer the question

3. CONTEXT PRECISION
   "Are the retrieved chunks actually relevant to the question?"
   Measures signal-to-noise in retrieval.
   Score 0-1. Low score = too many irrelevant chunks retrieved.
   
   Example:
   Retrieved 5 chunks, only 2 were used in the answer
   → Precision = 2/5 = 0.4

4. CONTEXT RECALL
   "Did we retrieve all the chunks needed to answer fully?"
   Score 0-1. Low score = missing information, answer is incomplete.
   
   Example:
   Answer needs 3 pieces of information, context only has 2
   → Recall = 2/3 = 0.67
""")

# Manual faithfulness check — implement from scratch
print("=" * 60)
print("Manual faithfulness calculation — from scratch")
print("=" * 60)

def check_faithfulness(answer, context_chunks):
    """
    Simplified faithfulness check.
    Real RAGAS uses an LLM to decompose the answer into claims
    and verify each claim against context.
    
    Here we approximate: does each sentence in the answer
    have semantic support in at least one chunk?
    """
    answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
    
    supported = 0
    for sentence in answer_sentences:
        sentence_embedding = model.encode(sentence)
        max_sim = max(
            cosine_sim(sentence_embedding, model.encode(chunk))
            for chunk in context_chunks
        )
        is_supported = max_sim > 0.5
        supported += int(is_supported)
        print(f"  Sentence: '{sentence[:60]}...'")
        print(f"  Max similarity to any chunk: {max_sim:.4f} → {'SUPPORTED' if is_supported else 'NOT SUPPORTED'}")
        print()
    
    faithfulness = supported / len(answer_sentences) if answer_sentences else 0
    return faithfulness


context = [
    "Backpropagation computes gradients by applying the chain rule recursively.",
    "The learning rate controls how large each weight update step is.",
]

# Faithful answer — everything traceable to context
faithful_answer = "Backpropagation applies the chain rule to compute gradients. The learning rate controls the size of weight updates."

# Hallucinated answer — contains facts not in context  
hallucinated_answer = "Backpropagation uses the chain rule. The learning rate should always be set to 0.001. Adam optimiser is the best choice."

print("FAITHFUL ANSWER:")
score1 = check_faithfulness(faithful_answer, context)
print(f"Faithfulness score: {score1:.2f}\n")

print("HALLUCINATED ANSWER:")
score2 = check_faithfulness(hallucinated_answer, context)
print(f"Faithfulness score: {score2:.2f}")

print("""
In production: RAGAS uses an LLM to decompose answers into atomic claims
and verify each one against context. More accurate than embedding similarity
but requires LLM calls per evaluation — run on a sample, not every query.
""")