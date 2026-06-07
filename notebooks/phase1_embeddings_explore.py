# =============================================================
# PHASE 1: Embeddings — understanding from first principles
# Run with: python notebooks/phase1_embeddings_explore.py
# =============================================================

from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------------------------------------
# BLOCK 1: What does an embedding actually look like?
# ------------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# First run downloads ~80MB model — cached after that

sentence = "The cat sat on the mat"
embedding = model.encode(sentence)

print("Type:", type(embedding))
print("Shape:", embedding.shape)        # How many dimensions?
print("First 5 values:", embedding[:5]) # What do the numbers look like?
print("Min:", embedding.min().round(4))
print("Max:", embedding.max().round(4))

# ------------------------------------------------------------
# BLOCK 2: Cosine similarity — do similar sentences cluster together?
# ------------------------------------------------------------

def cosine_similarity(a, b):
    # Implementing manually so you understand it — not using a library
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

sentences = {
    "A": "The cat sat on the mat",
    "B": "A feline rested on the rug",       # semantically similar to A
    "C": "Dogs love to play in the park",    # animals but different meaning
    "D": "Quantum entanglement in physics",  # completely unrelated
    "E": "The kitten lay on the carpet",     # very close to A
}

embeddings = {k: model.encode(v) for k, v in sentences.items()}

print("\n--- Cosine Similarity Matrix ---")
print(f"{'':>4}", end="")
for k in sentences:
    print(f"  {k}    ", end="")
print()

for k1 in sentences:
    print(f"{k1:>4}", end="")
    for k2 in sentences:
        sim = cosine_similarity(embeddings[k1], embeddings[k2])
        print(f"  {sim:.3f}", end="")
    print()

print("\n--- Key pairs to notice ---")
pairs = [("A","B"), ("A","E"), ("A","C"), ("A","D"), ("B","E")]
for p1, p2 in pairs:
    sim = cosine_similarity(embeddings[p1], embeddings[p2])
    print(f"  {sentences[p1]!r:45s} vs {sentences[p2]!r}")
    print(f"  Similarity: {sim:.4f}\n")

# ------------------------------------------------------------
# BLOCK 3: Why cosine similarity, not Euclidean distance?
# ------------------------------------------------------------

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# A short and long version of essentially the same meaning
short = "Cats are pets"
long  = "Cats are wonderful domestic pets that have been companions to humans for thousands of years"

v_short = model.encode(short)
v_long  = model.encode(long)

cos_sim  = cosine_similarity(v_short, v_long)
euc_dist = euclidean_distance(v_short, v_long)

print(f"Short: '{short}'")
print(f"Long:  '{long[:60]}...'")
print(f"\nCosine similarity : {cos_sim:.4f}  ← high = same meaning")
print(f"Euclidean distance: {euc_dist:.4f}  ← what does this tell us?")

# Now compare with something unrelated but also short
unrelated_short = "Quantum entanglement"
v_unrelated = model.encode(unrelated_short)

cos_unrelated = cosine_similarity(v_short, v_unrelated)
euc_unrelated = euclidean_distance(v_short, v_unrelated)

print(f"\nUnrelated short: '{unrelated_short}'")
print(f"Cosine similarity : {cos_unrelated:.4f}")
print(f"Euclidean distance: {euc_unrelated:.4f}")

print("\n--- Key insight ---")
print("Cosine measures ANGLE between vectors (direction = meaning)")
print("Euclidean measures LENGTH difference (affected by sentence length)")
print("Two sentences can be far apart in L2 but point in the same direction.")

# ------------------------------------------------------------
# BLOCK 4: The RAG simulation — this is exactly what retrieval does
# ------------------------------------------------------------

# Imagine these are chunks stored in your vector store
document_chunks = [
    "The transformer architecture uses self-attention mechanisms.",
    "BERT is a bidirectional transformer trained on masked language modelling.",
    "Gradient descent optimises neural network weights by minimising loss.",
    "Photosynthesis is the process by which plants convert sunlight to energy.",
    "The Eiffel Tower is located in Paris, France.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Backpropagation computes gradients by the chain rule.",
    "The Seine river flows through the city of Paris.",
]

# Embed all chunks — this is what happens during ingestion
print("Embedding document chunks...")
chunk_embeddings = model.encode(document_chunks)

def retrieve(query, top_k=3):
    query_embedding = model.encode(query)
    
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, chunk_emb)
        similarities.append((sim, i, document_chunks[i]))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} retrieved chunks:")
    for rank, (sim, idx, chunk) in enumerate(similarities[:top_k], 1):
        print(f"  [{rank}] score={sim:.4f} | {chunk}")

# Test with different queries
retrieve("How does attention work in neural networks?")
retrieve("Tell me about Paris")
retrieve("How are model parameters updated during training?")