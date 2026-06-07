# =============================================================
# PHASE 3: Vector Stores — storing and retrieving embeddings
# Run with: python notebooks\phase3_vectorstore_explore.py
# =============================================================

# ------------------------------------------------------------
# BLOCK 1: Build a naive vector store from scratch
# Before using ChromaDB, understand what it's doing underneath
# ------------------------------------------------------------

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# This is conceptually what a vector store is —
# a list of (id, embedding, text, metadata) tuples
naive_store = []

def naive_add(chunk_id, text, metadata):
    embedding = model.encode(text)
    naive_store.append({
        "id": chunk_id,
        "embedding": embedding,
        "text": text,
        "metadata": metadata,
    })

def naive_query(query, top_k=3):
    query_embedding = model.encode(query)
    
    scores = []
    for item in naive_store:
        dot = np.dot(query_embedding, item["embedding"])
        norm = np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
        similarity = dot / norm
        scores.append((similarity, item))
    
    # Sort by similarity
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# Add some chunks
chunks = [
    ("doc1_p1_c1", "Neural networks learn by adjusting weights through backpropagation.", {"source": "dl_textbook.pdf", "page": 1}),
    ("doc1_p1_c2", "The transformer architecture relies on self-attention mechanisms.", {"source": "dl_textbook.pdf", "page": 2}),
    ("doc1_p2_c1", "Gradient descent minimises the loss function iteratively.", {"source": "dl_textbook.pdf", "page": 3}),
    ("doc2_p1_c1", "Mumbai is the financial capital of India.", {"source": "india_facts.pdf", "page": 1}),
    ("doc2_p1_c2", "The Ganges is one of the most sacred rivers in India.", {"source": "india_facts.pdf", "page": 2}),
    ("doc2_p1_c3", "Bangalore is known as the Silicon Valley of India.", {"source": "india_facts.pdf", "page": 3}),
]

print("Adding chunks to naive store...")
for chunk_id, text, metadata in chunks:
    naive_add(chunk_id, text, metadata)
print(f"Store contains {len(naive_store)} chunks\n")

# Query it
print("=" * 60)
print("QUERY: 'How do neural networks learn?'")
print("=" * 60)
results = naive_query("How do neural networks learn?", top_k=3)
for sim, item in results:
    print(f"  score={sim:.4f} | [{item['metadata']['source']} p{item['metadata']['page']}]")
    print(f"  {item['text']}\n")

print("=" * 60)
print("QUERY: 'Tell me about Indian cities'")
print("=" * 60)
results = naive_query("Tell me about Indian cities", top_k=3)
for sim, item in results:
    print(f"  score={sim:.4f} | [{item['metadata']['source']} p{item['metadata']['page']}]")
    print(f"  {item['text']}\n")

print("=" * 60)
print("THE PROBLEM WITH THIS NAIVE STORE")
print("=" * 60)
print(f"Chunks in store: {len(naive_store)}")
print("Search method: brute force — computes similarity against EVERY chunk")
print("At 1000 chunks: 1000 comparisons per query")
print("At 1M chunks: 1,000,000 comparisons per query — unusable latency")
print("\nThis is exactly why vector databases exist.")
print("ChromaDB uses HNSW indexing — O(log n) instead of O(n)")

# ------------------------------------------------------------
# BLOCK 2: ChromaDB — persistent vector store with HNSW indexing
# This is what src/retrieval/vector_store.py wraps
# ------------------------------------------------------------

import chromadb
from chromadb.config import Settings

print("\n" + "=" * 60)
print("BLOCK 2: ChromaDB — persistent vector store")
print("=" * 60)

# PersistentClient saves to disk — survives between runs
# This is the key difference from our naive in-memory store
client = chromadb.PersistentClient(
    path="./chroma_db_explore",
    settings=Settings(anonymized_telemetry=False),
)

# Delete and recreate for clean experiment
try:
    client.delete_collection("phase3_explore")
except:
    pass

collection = client.get_or_create_collection(
    name="phase3_explore",
    metadata={"hnsw:space": "cosine"},  # Tell Chroma to use cosine distance
)

# Add the same chunks — but now Chroma handles embedding storage and indexing
texts = [c[1] for c in chunks]
ids = [c[0] for c in chunks]
metadatas = [c[2] for c in chunks]
embeddings = model.encode(texts).tolist()

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
)

print(f"Collection '{collection.name}' contains {collection.count()} chunks")
print("Persisted to ./chroma_db_explore — this survives process restarts\n")

# Query ChromaDB
def chroma_query(query_text, top_k=3, filter_source=None):
    query_embedding = model.encode(query_text).tolist()
    
    where_filter = {"source": filter_source} if filter_source else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )
    
    print(f"Query: '{query_text}'")
    if filter_source:
        print(f"Filter: source = '{filter_source}'")
    
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)  # Chroma returns distance, not similarity
        print(f"  score={similarity:.4f} | [{meta['source']} p{meta['page']}]")
        print(f"  {doc}\n")


print("=" * 60)
print("SAME QUERY — now through ChromaDB")
print("=" * 60)
chroma_query("How do neural networks learn?")

print("=" * 60)
print("METADATA FILTERING — only search within one document")
print("=" * 60)
chroma_query(
    "Tell me about India",
    filter_source="india_facts.pdf"
)

print("=" * 60)
print("THE POWER OF METADATA FILTERING")
print("=" * 60)
print("""
In a real RAG system with 50 PDFs ingested:
- Without filter: query searches all 50,000 chunks
- With filter:    query searches only chunks from the selected PDF

Use case: "Search only within this uploaded document"
This is how multi-tenant RAG systems isolate user data.
A filter on user_id metadata means User A never retrieves User B's chunks.
""")

# ------------------------------------------------------------
# BLOCK 3: What breaks in a vector store — failure modes to know
# ------------------------------------------------------------

print("=" * 60)
print("BLOCK 3: Failure modes — what breaks and why")
print("=" * 60)

# Failure 1: Querying with a different embedding model than ingestion
print("FAILURE 1: Embedding model mismatch")
print("-" * 40)

from sentence_transformers import SentenceTransformer
model_different = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Query with different model — chunks were embedded with MiniLM
wrong_query_embedding = model_different.encode("How do neural networks learn?").tolist()

print("Chunks embedded with: all-MiniLM-L6-v2 (384 dims)")
print("Query embedded with:  all-mpnet-base-v2 (768 dims)")
print()

try:
    results = collection.query(
        query_embeddings=[wrong_query_embedding],
        n_results=3,
        include=["documents", "distances"],
    )
    print("Results:", results["documents"][0])
except Exception as e:
    print(f"Chroma caught it: {type(e).__name__}")
    print(f"Message: {e}")
    print("""
Key insight: query and document embeddings MUST use the same model.
If you switch embedding models, you must re-embed ALL chunks from scratch.
The dangerous case: two models with same output dimension but different
architecture — Chroma won't catch it, scores will be silently wrong.
""")

# Failure 2: Duplicate chunk IDs
print("FAILURE 2: Duplicate IDs — what happens?")
print("-" * 40)
try:
    collection.add(
        ids=["doc1_p1_c1"],
        embeddings=[model.encode("Some different text").tolist()],
        documents=["Some different text"],
        metadatas=[{"source": "new.pdf", "page": 1}],
    )
    print("Added duplicate — Chroma silently ignored or overwrote")
except Exception as e:
    print(f"Chroma raised: {type(e).__name__}")
    print("Fix: use collection.upsert() instead of add() when re-ingesting")

print(f"\nCollection count after duplicate attempt: {collection.count()}")

# Failure 3: Empty query
print("\nFAILURE 3: What does an empty query retrieve?")
print("-" * 40)
empty_embedding = model.encode("").tolist()
results = collection.query(
    query_embeddings=[empty_embedding],
    n_results=2,
    include=["documents", "distances"],
)
print("Query: '' (empty string)")
print("Retrieved:", results["documents"][0])
print("Lesson: always validate query is non-empty before hitting the vector store.")

# Cleanup
client.delete_collection("phase3_explore")
print("\nExplore collection cleaned up.")