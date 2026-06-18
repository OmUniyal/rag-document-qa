# RAG Document Q&A

A production-structured Retrieval-Augmented Generation (RAG) system built from scratch.
Ask natural language questions against your own PDF documents.

## Live Demo
👉 [Try it on HuggingFace Spaces](https://huggingface.co/spaces/omUniyal/rag-document-qa)

Upload any PDF and ask questions about it instantly. Pre-loaded with the Attention Is All You Need and RAG papers.

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
| LLM | Groq (Llama 3.1 8B) | Free tier, fast inference, no local install |
| UI | Gradio | One-file deploy to HF Spaces |

## Setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/OmUniyal/rag-document-qa
cd rag-document-qa
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Add your GROQ_API_KEY to .env (get a free key at console.groq.com)

# 4. Drop your PDFs into data/raw/

# 5. Ingest
python ingest.py

# 6. Run the app
python app.py
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
