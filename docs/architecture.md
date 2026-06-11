# RAG Document Q&A — Architecture & Build Journal

This document explains how the system works, why each design decision was made, and what was learned building it. It is written for someone who wants to understand the project deeply, not just run it.

---

## What This Project Is

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about PDF documents. Built from first principles — every component implemented and understood before any abstraction layer was added.

**The core idea:** Instead of asking an LLM to answer from its training data (which has a knowledge cutoff and can hallucinate), RAG retrieves relevant passages from your own documents and instructs the LLM to answer only from those passages. The LLM provides reasoning. Your documents provide facts.

---

## System Architecture

```
PDF Files
   │
   ▼
[pdf_loader.py]          Extract text page by page, preserve page numbers
   │
   ▼
[chunker.py]             Split pages into overlapping chunks
   │
   ▼
[embedder.py]            Convert chunks to 384-dim vectors (sentence-transformers)
   │
   ▼
[vector_store.py]        Store vectors + metadata in ChromaDB (persisted to disk)


At query time:

User Question
   │
   ▼
[embedder.py]            Embed the question using the same model
   │
   ▼
[vector_store.py]        Find top-k most similar chunks (cosine similarity)
   │
   ▼
[prompt_builder.py]      Build grounded prompt: system instruction + context + question
   │
   ▼
[llm_client.py]          Send to Groq API (Llama 3.1 8B)
   │
   ▼
Answer + Source Citations
```

---

## Component Deep Dives

### 1. Embeddings (`src/retrieval/embedder.py`)

**What:** Converts text into a 384-dimensional float vector using `sentence-transformers/all-MiniLM-L6-v2`.

**Why sentence-transformers:** The model is a fine-tuned BERT-style encoder trained on sentence similarity tasks. It produces vectors where semantically similar sentences point in the same direction in 384-dimensional space — regardless of whether they share vocabulary. "The cat sat on the mat" and "A feline rested on the rug" produce similar vectors despite sharing zero words.

**Why cosine similarity, not Euclidean distance:** Cosine measures the angle between vectors, not their magnitude. Two sentences meaning the same thing — one short, one long — will point in the same direction but may differ in magnitude. Cosine handles this correctly; Euclidean distance would penalise the magnitude difference.

**Key insight from experimentation:** Embedding models capture usage patterns, not dictionary definitions. During training on billions of sentences, the model learned that "cat" and "feline" appear in identical contexts repeatedly — so their vectors end up pointing in similar directions.

**Interview question this answers:** *"What is a sentence embedding and why do we use cosine similarity for semantic search?"*

---

### 2. Chunking (`src/ingestion/chunker.py`)

**What:** Splits raw PDF text into overlapping chunks of ~500 characters.

**Why chunk at all:** LLMs have token limits. A 100-page PDF cannot be fed directly. Chunking breaks documents into retrievable units.

**Why overlap:** A sentence split across chunk boundaries would be missed by retrieval. Overlap ensures boundary content appears fully in at least one chunk. Cost: ~33% more storage and embedding compute.

**The fixed-size problem:** Character-based chunking cuts mid-word and mid-sentence. Observed in experiments: "connectionist" became "connec" in one chunk and "tionist" in the next. Neither chunk would score well for a query about "connectionist approaches."

**Sentence-aware chunking:** Splitting at sentence boundaries prevents mid-word cuts. The chunker in this project uses fixed-size with overlap as a baseline. Production systems would use sentence-aware or structure-aware chunking (splitting at section headers).

**Chunk size tradeoff:**
- Smaller chunks → more precise retrieval, but cross-sentence context is lost
- Larger chunks → better context, but lower retrieval precision

**Interview question this answers:** *"What chunking strategy did you use and why? What are the tradeoffs?"*

---

### 3. Vector Store (`src/retrieval/vector_store.py`)

**What:** ChromaDB persistent vector store using HNSW indexing with cosine distance.

**Why ChromaDB:** Runs locally with no infrastructure. Persists to disk — vectors survive process restarts. No re-embedding on every run.

**How HNSW works:** Brute-force cosine search is O(n) — unusable at 1M+ chunks. HNSW (Hierarchical Navigable Small World) builds a multi-layer graph:
- Top layers: few nodes, long connections — fast coarse navigation
- Bottom layer: all nodes, short connections — precise local search

Query time navigates top-down, narrowing to the closest neighbourhood, then searches precisely within it. Result: O(log n) instead of O(n). Tradeoff: approximate — may miss the single closest vector by a small margin, but recall stays above 95% in practice.

**Metadata filtering:** Every chunk stores `source` (filename) and `page` (page number) as metadata. Queries can be filtered to a specific document — essential for multi-document systems where User A's documents should not mix with User B's.

**Failure modes observed:**
1. **Model mismatch:** Querying with a 768-dim vector against a 384-dim collection raises `InvalidArgumentError`. If two models have the same dimension but different architectures, Chroma won't catch it — scores will be silently wrong.
2. **Duplicate IDs:** `collection.add()` with an existing ID silently ignores the new data. Use `collection.upsert()` for re-ingestion workflows.
3. **Empty query:** An empty string produces a valid embedding vector. Chroma returns results confidently. Always validate query input before hitting the vector store.

**Interview question this answers:** *"What is HNSW and why does it matter at scale? What breaks in a vector store?"*

---

### 4. Prompt Engineering (`src/generation/prompt_builder.py`)

**What:** Constructs the grounded prompt sent to the LLM.

**The naive approach (wrong):**
```
Answer this question: {query}
```
The LLM answers from training data. No grounding. No source attribution. This is a chatbot, not a RAG system.

**The grounded approach (correct):**
```
You are a precise document assistant. Answer ONLY using the context below.
If the answer is not in the context, say: "I could not find this in the provided documents."
Always cite which source and page your answer comes from.

=== CONTEXT ===
[Source 1: document.pdf, Page 4]
{chunk text}

[Source 2: document.pdf, Page 5]
{chunk text}

=== QUESTION ===
{query}

=== ANSWER ===
```

**Why each part exists:**
1. `Answer ONLY using the context below` — restricts the LLM to the retrieved context
2. `I could not find this...` — explicit fallback prevents hallucination when context is irrelevant
3. Source labels — enable citation in the answer
4. `=== CONTEXT ===` separators — make structure unambiguous to the model

**The system prompt is the contract between RAG and the LLM.** Without it, the retrieval layer is ignored entirely.

**Temperature:** Set to 0.1 (low). In a RAG system, low temperature produces consistent, factual answers. High temperature increases hallucination risk — the model starts sampling less probable tokens not grounded in the retrieved text.

**Interview question this answers:** *"How do you prevent hallucination in a RAG system?"*

---

### 5. LLM Client (`src/generation/llm_client.py`)

**What:** Groq API client using `llama-3.1-8b-instant`.

**Why Groq over OpenAI:** Free tier with generous rate limits (14,400 requests/day). Fast inference. Avoids vendor lock-in — the `LLMClient` class abstracts the backend so swapping to a different provider requires changing one file.

**Why not fine-tune:** RAG and fine-tuning solve different problems. Fine-tuning bakes knowledge into model weights — expensive, slow to update, requires retraining when documents change. RAG keeps knowledge in the vector store — cheap, updatable in real time by re-ingesting documents. For document Q&A, RAG is almost always the right choice.

**Interview question this answers:** *"What is the difference between RAG and fine-tuning? When would you use each?"*

---

### 6. Advanced Retrieval (`notebooks/phase5_advanced_retrieval.py`)

Three retrieval improvements explored beyond basic dense retrieval:

**Hybrid Search (dense + sparse):**
Dense retrieval (embeddings) captures semantic similarity but misses exact keyword matches. BM25 sparse retrieval rewards exact keyword matches but misses semantic similarity. Hybrid combines both with a weighted score:

```
combined = alpha * dense_norm + (1 - alpha) * sparse_norm
```

Limitation observed: When vocabulary mismatch is complete (query and chunks share zero words), BM25 scores everything zero and hybrid degenerates to pure dense retrieval.

**Re-ranking with Cross-Encoders:**
Two-stage approach:
1. Retrieve top-20 candidates fast using bi-encoder (dense retrieval)
2. Re-rank using cross-encoder — reads query and chunk *together* in one forward pass

Why cross-encoders are more accurate: bi-encoders embed query and chunk *separately*, then compare vectors — the query and chunk never interact during encoding. Cross-encoders see the full interaction between query and chunk tokens via attention, producing more accurate relevance scores.

Why not use cross-encoders for full retrieval: too slow. Must run a full forward pass for every query-chunk pair at query time — unusable for 100,000 chunks, fine for 20 candidates.

**RAGAS Evaluation:**
Four metrics for measuring RAG quality:

| Metric | Measures | Low score means |
|---|---|---|
| Faithfulness | Is the answer supported by context? | Hallucination |
| Answer Relevancy | Does the answer address the question? | Off-topic response |
| Context Precision | Are retrieved chunks relevant? | Too much noise |
| Context Recall | Were all needed chunks retrieved? | Incomplete answer |

**Precision vs recall tradeoff:** For general use, sacrifice recall over precision. An incomplete honest answer ("I could not find this") is recoverable. A confidently wrong answer erodes trust permanently. Exception: medical/legal RAG where missing critical information is the greater risk.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| No LangChain in core modules | Implemented each component directly | Interviewers probe whether you understand what abstractions do |
| Config-driven parameters | `configs/config.yaml` | No magic strings scattered in source files |
| Groq over OpenAI | `llama-3.1-8b-instant` | Free tier, no vendor lock-in |
| pdfplumber over PyMuPDF | Pure Python | PyMuPDF DLL blocked by application control policy |
| sentence-transformers local | `all-MiniLM-L6-v2` | No API cost, runs offline, full control |
| Persistent ChromaDB | `./chroma_db` | Survives restarts — no re-embedding on every run |

---

## Failure Modes & What They Teach

| Failure | Root Cause | Fix |
|---|---|---|
| Wrong chunks ranked first | Vocabulary mismatch — query and chunk use different words for same concept | Hybrid search or re-ranking |
| Noise chunks in top-k | top-k always returns k results regardless of quality | Similarity threshold filter |
| Hallucination | System prompt missing restriction instruction | Add "Answer ONLY from context" + fallback |
| Model mismatch error | Query embedded with different model than documents | Always use same embedding model for ingestion and query |
| Stale chunks after re-ingestion | `collection.add()` ignores duplicate IDs | Use `collection.upsert()` |
| Empty query returns results | Empty string produces valid embedding vector | Validate query before hitting vector store |

---

## How to Run

```bash
# 1. Clone and set up environment
git clone <repo>
cd rag-document-qa
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 3. Ingest a PDF
python ingest.py --file data/raw/your_document.pdf

# 4. Run the app
python app.py

# 5. Run tests
pytest tests/unit/ -v
```

---

## Project Structure

```
rag-document-qa/
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF → pages (pdfplumber)
│   │   └── chunker.py         # Pages → overlapping chunks
│   ├── retrieval/
│   │   ├── embedder.py        # Text → 384-dim vectors
│   │   ├── vector_store.py    # ChromaDB wrapper (HNSW indexing)
│   │   └── retriever.py       # Embed query → search store → return chunks
│   ├── generation/
│   │   ├── prompt_builder.py  # Build grounded prompt with context + citations
│   │   ├── llm_client.py      # Groq API wrapper
│   │   └── rag_chain.py       # Full pipeline: retrieve → prompt → generate
│   ├── evaluation/
│   │   └── ragas_eval.py      # RAGAS metrics (faithfulness, precision, recall)
│   └── utils/
│       ├── config.py          # Loads config.yaml + .env into typed object
│       └── logger.py          # Centralised loguru logger
├── notebooks/
│   ├── phase1_embeddings_explore.py      # Cosine similarity, embedding visualisation
│   ├── phase2_chunking_explore.py        # Fixed-size vs sentence-aware chunking
│   ├── phase3_vectorstore_explore.py     # Naive store → ChromaDB → failure modes
│   ├── phase4_generation_explore.py      # Prompt engineering, grounding, temperature
│   └── phase5_advanced_retrieval.py      # Hybrid search, re-ranking, RAGAS
├── configs/
│   └── config.yaml            # All tuneable parameters
├── data/raw/                  # Drop PDFs here (gitignored)
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   └── test_embedder.py
│   └── integration/
│       └── test_rag_pipeline.py
├── ingest.py                  # CLI: PDF → chunks → embeddings → ChromaDB
├── app.py                     # Gradio UI
├── .env.example               # Template for secrets
└── requirements.txt           # Direct dependencies only
```

---

## What Each Exploration Notebook Proves

| Notebook | Key finding |
|---|---|
| phase1 | "The cat sat on the mat" and "A feline rested on the rug" score 0.56 similarity — zero shared words |
| phase2 | Fixed-size chunking cuts "connectionist" into "connec" + "tionist" — neither chunk retrieves correctly |
| phase3 | ChromaDB scores exactly match naive brute-force cosine — HNSW is correct, just faster |
| phase4 | Without system prompt restriction, LLM ignores retrieved context entirely |
| phase5 | Vocabulary mismatch makes BM25 score everything zero — hybrid collapses to pure dense |

---

*Built phase by phase, June 2026.*
