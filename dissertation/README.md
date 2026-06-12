# Dissertation Notes & Planning

This folder tracks the evolution of the RAG project toward a potential
BITS WILP M.Tech dissertation submission.

Updated as the project grows — ideas, paper references, extension plans,
and organizational use cases are documented here in real time.

---

## Project Overview

**Working title:** Domain-Specific RAG System for Enterprise Document Intelligence

**Core system:** A Retrieval-Augmented Generation pipeline that answers
natural language questions over PDF documents. Built from first principles
— embeddings, chunking, vector storage, grounded generation, and evaluation
implemented without black-box abstractions.

**GitHub:** rag-document-qa

**Started:** June 2026

---

## Potential Dissertation Angles

These are directions the project could grow into a novel academic contribution.
Not all will be pursued — these are options to evaluate over the next semester.

### Option 1: Multimodal RAG (Preferred)
Extend the system to handle PDFs containing images, charts, tables, and
diagrams alongside text. Most enterprise documents are mixed-content.

**Novel contribution:**
- Comparative study of text-only vs multimodal retrieval quality on
  enterprise documents
- CLIP-based image embedding pipeline integrated with existing text pipeline
- Evaluation framework for multimodal retrieval (no standard benchmark exists
  for enterprise mixed-content documents)

**Aligns with:** Multimodal Information Retrieval course (Sem 3)

**Organizational relevance:** Marketing and campaign documents at work
contain charts, brand guidelines with images, and mixed-content PDFs.
A multimodal RAG system could answer questions across all of these.

---

### Option 2: GraphRAG — Knowledge Graph Enhanced Retrieval
Replace flat vector similarity with a knowledge graph layer. Extract
entities and relationships from documents, build a graph, traverse it
during retrieval.

**Novel contribution:**
- Comparison of flat RAG vs GraphRAG on a domain-specific corpus
- Leverages relational thinking from Oracle SQL/database background
- Graph structure captures document relationships flat chunking misses

**Aligns with:** Graph Neural Networks (available Sem 3, not selected)

**Organizational relevance:** Data pipelines at work have complex entity
relationships — campaigns, users, segments, attributes. A graph-based
system could represent and query these relationships more naturally.

---

### Option 3: Multilingual RAG for Enterprise Campaigns
Extend the RAG system to handle documents and queries in multiple languages.
Relevant to multilingual marketing campaign workflows.

**Novel contribution:**
- Cross-lingual retrieval — query in English, retrieve from Hindi/regional
  language documents or vice versa
- Evaluation of multilingual embedding models on enterprise campaign content
- Integration with existing MoEngage multilingual campaign architecture

**Aligns with:** NLP Applications course (Sem 3), existing work experience

**Organizational relevance:** Direct — multilingual campaign flows already
implemented at work. A RAG layer that can answer questions across
multilingual campaign documents would be immediately useful.

---

### Option 4: RAG Evaluation Framework
Build a rigorous, domain-specific evaluation framework comparing chunking
strategies, embedding models, retrieval methods, and generation quality.

**Novel contribution:**
- Standardised evaluation methodology for enterprise RAG systems
- Dataset construction methodology for domain-specific Q&A evaluation
- Benchmark results across multiple RAG configurations

**Aligns with:** All four Sem 3 courses indirectly

---

## Organizational Use Cases

The RAG project has direct applicability to current work. These use cases
could form the basis of a dissertation that satisfies both academic and
organizational requirements.

### Use Case 1: Campaign Intelligence Assistant
**Problem:** Marketing teams spend significant time searching through past
campaign documents, performance reports, and brand guidelines to answer
questions like "What subject lines worked best for re-engagement campaigns?"

**RAG solution:** Ingest campaign reports, performance PDFs, and strategy
documents. Enable natural language queries across the entire campaign history.

**Data sources:** Campaign PDFs, performance reports, brand guidelines,
strategy documents

---

### Use Case 2: Data Pipeline Documentation Assistant
**Problem:** SnapLogic pipelines and in-house orchestration tools generate
fragmented documentation. New team members struggle to understand pipeline
logic, and even experienced members lose time hunting for specific pipeline
behaviour.

**RAG solution:** Ingest pipeline documentation, SQL query files, and
architecture documents. Answer questions like "Which pipeline handles user
attribute sync?" or "What does this transformation step do?"

**Data sources:** Pipeline docs, SQL files, architecture diagrams (text),
runbooks

**Note:** This directly extends the pipeline observability tool concept
explored as a hackathon entry.

---

### Use Case 3: MoEngage Feature Documentation Q&A
**Problem:** Product and marketing teams frequently ask repetitive questions
about MoEngage features, API capabilities, and integration options that are
documented but hard to search.

**RAG solution:** Ingest MoEngage documentation, release notes, and internal
how-to guides. Enable teams to ask questions in natural language rather than
searching through documentation manually.

---

## Technical Foundation Already Built

These components are already implemented and tested — dissertation builds on top:

| Component | Status | Location |
|---|---|---|
| PDF ingestion | Complete | src/ingestion/pdf_loader.py |
| Text chunking | Complete | src/ingestion/chunker.py |
| Dense embeddings | Complete | src/retrieval/embedder.py |
| ChromaDB vector store | Complete | src/retrieval/vector_store.py |
| Hybrid search (BM25 + dense) | Explored | notebooks/phase5_advanced_retrieval.py |
| Re-ranking (cross-encoder) | Explored | notebooks/phase5_advanced_retrieval.py |
| Grounded prompt engineering | Complete | src/generation/prompt_builder.py |
| LLM generation (Groq) | Complete | src/generation/llm_client.py |
| Full RAG chain | Complete | src/generation/rag_chain.py |
| RAGAS evaluation concepts | Explored | notebooks/phase5_advanced_retrieval.py |
| Unit + integration tests | Complete | tests/ |
| Architecture documentation | Complete | docs/architecture.md |

---

## Papers to Read

These are foundational papers worth reading before the dissertation phase.
Added as they become relevant.

- **Attention Is All You Need** (Vaswani et al., 2017) — transformer architecture
- **BERT** (Devlin et al., 2018) — bidirectional encoders, your embedding model's foundation
- **Dense Passage Retrieval** (Karpukhin et al., 2020) — bi-encoder retrieval
- **RAG** (Lewis et al., 2020) — the original RAG paper
- **RAGAS** (Es et al., 2023) — evaluation framework you implemented
- **GraphRAG** (Edge et al., 2024, Microsoft) — knowledge graph enhanced RAG
- **CLIP** (Radford et al., 2021) — multimodal embeddings for Option 1
- **ColBERT** (Khattab & Zaharia, 2020) — late interaction retrieval, advanced re-ranking

---

## Timeline (Rough)

| Period | Goal |
|---|---|
| Now — Sem 3 start | Solidify current RAG project, add document type support |
| Sem 3 | Extend toward chosen dissertation direction (likely multimodal) |
| Sem 3 end | Identify dissertation supervisor, confirm topic |
| Dissertation sem | Full implementation, evaluation, and write-up |

---

## Notes & Ideas

*Add raw thoughts here as they come — no structure needed.*

- June 2026: Project started as portfolio piece for GenAI/LLM engineering roles.
  Grew into potential dissertation candidate naturally.
- Oracle SQL background connects to GraphRAG direction — relational thinking
  maps to graph traversal intuitively.
- Multilingual RAG connects to existing MoEngage work — could satisfy
  organizational relevance requirement if BITS WILP expects industry applicability.
- Pipeline observability tool concept (hackathon idea) + RAG = Use Case 2 above.
  Two ideas that were separate now converge.
