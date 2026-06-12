"""
Gradio UI — RAG Document Q&A
Deployed on HuggingFace Spaces.

Pre-ingested documents:
  - Attention Is All You Need (Vaswani et al., 2017)
  - RAG for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
"""

import os
from pathlib import Path
import gradio as gr
from src.generation.rag_chain import RAGChain
from src.retrieval.vector_store import VectorStore
from src.ingestion.pdf_loader import load_pdfs_from_dir
from src.ingestion.chunker import chunk_pages
from src.retrieval.embedder import Embedder
from src.utils.logger import logger


def ensure_ingested():
    """
    Auto-ingest PDFs on first startup if vector store is empty.
    On HuggingFace Spaces, the chroma_db doesn't persist between restarts
    so we re-ingest from data/raw/ every cold start.
    """
    store = VectorStore()
    if store.collection.count() > 0:
        logger.info(f"Vector store already has {store.collection.count()} chunks — skipping ingestion")
        return

    logger.info("Vector store empty — ingesting PDFs from data/raw/...")
    pages = load_pdfs_from_dir("data/raw")
    if not pages:
        logger.error("No PDFs found in data/raw/ — app will not work correctly")
        return

    chunks = chunk_pages(pages)
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(chunks, embeddings)
    logger.info(f"Ingestion complete — {len(chunks)} chunks stored")


# Run ingestion on startup
ensure_ingested()

# Initialise RAG chain
chain = RAGChain()


def answer_question(question: str) -> tuple[str, str]:
    """Gradio callback: takes question, returns (answer, sources_markdown)."""
    if not question.strip():
        return "Please enter a question.", ""

    result = chain.query(question)
    answer = result["answer"]

    sources_lines = ["**Sources retrieved:**\n"]
    for i, s in enumerate(result["sources"], start=1):
        sources_lines.append(
            f"{i}. `{s['source']}` — Page {s['page']} (similarity: {s['score']})"
        )
    sources_md = "\n".join(sources_lines)
    return answer, sources_md


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------

with gr.Blocks(
    title="RAG Document Q&A",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # RAG Document Q&A
    Ask questions about the **Attention Is All You Need** and **RAG** papers.
    Answers are grounded only in the retrieved document passages — no hallucination from training data.
    """)

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the attention mechanism? How does RAG work?",
            lines=2,
        )

    submit_btn = gr.Button("Ask", variant="primary", size="lg")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False,
        )
        sources_box = gr.Markdown(label="Sources")

    gr.Examples(
        examples=[
            "What is the attention mechanism in transformers?",
            "What is multi-head attention?",
            "How does RAG combine retrieval and generation?",
            "What datasets were used to evaluate RAG?",
            "What is the encoder-decoder architecture?",
        ],
        inputs=question_box,
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box],
        outputs=[answer_box, sources_box],
    )

    gr.Markdown(
        "_Built with sentence-transformers, ChromaDB, and Groq (Llama 3.1 8B). "
        "[View source on GitHub](https://github.com/OmUniyal/rag-document-qa)_"
    )

if __name__ == "__main__":
    demo.launch()