# RAG Document Q&A

A production-structured Retrieval-Augmented Generation (RAG) system built from scratch.
Ask natural language questions against your own PDF documents.

## Architecture

```
PDF → [pdf_loader] → [chunker] → [embedder] → [vector_store (ChromaDB)]
                                                        ↓
Query → [embedder] → [vector_store.query()] → [prompt_builder] → [LLM] → Answer + Sources
```

## Stack

| Layer | Tool | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, accurate text extraction |
| Embeddings | sentence-transformers | Local, no API cost |
| Vector store | ChromaDB | Persistent, no infra needed |
| LLM | Ollama (Mistral 7B) | Local, private, free |
| UI | Gradio | One-file deploy to HF Spaces |

## Setup

```bash
# 1. Clone and create virtual environment
git clone <your-repo>
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
make install

# 3. Configure
cp .env.example .env
# Edit .env if using HuggingFace backend instead of Ollama

# 4. Install and start Ollama (for local LLM)
# Download from https://ollama.ai
ollama pull mistral

# 5. Drop your PDFs into data/raw/
# 6. Ingest
make ingest

# 7. Run the app
make run
```

## Development

```bash
make test       # Run unit tests
make lint       # Check code style
make format     # Auto-format with black
make ingest-reset  # Clear vector store and re-ingest
```

## Design decisions

- **No LangChain in core modules**: each step (chunker, embedder, vector_store, prompt_builder)
  is implemented directly so the pipeline is fully understandable and debuggable.
- **Config-driven**: all tuneable parameters live in `configs/config.yaml`, not scattered as
  magic strings in source files.
- **Source attribution**: every answer includes which document and page it came from.
- **Backend-agnostic LLM**: switch between Ollama (local) and HuggingFace API via config,
  no code changes required.

## Evaluation

RAGAS metrics tracked: faithfulness, answer relevancy, context precision, context recall.
See `src/evaluation/ragas_eval.py` and `notebooks/03_retrieval_eval.ipynb`.
