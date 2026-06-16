"""
Gradio UI — RAG Document Q&A
Deployed on HuggingFace Spaces.

Supports:
  - Pre-loaded demo documents (Attention Is All You Need, RAG paper)
  - User PDF uploads — upload any PDF and query it instantly
"""

import os
import tempfile
from pathlib import Path
import gradio as gr
from src.generation.rag_chain import RAGChain
from src.retrieval.vector_store import VectorStore
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.retrieval.embedder import Embedder
from src.utils.logger import logger


# ------------------------------------------------------------
# Startup — ingest demo documents if store is empty
# ------------------------------------------------------------

def ensure_demo_ingested():
    store = VectorStore()
    if store.collection.count() > 0:
        logger.info(f"Vector store has {store.collection.count()} chunks — skipping demo ingestion")
        return store.collection.count()

    logger.info("Ingesting demo documents...")
    from src.ingestion.pdf_loader import load_pdfs_from_dir
    from src.ingestion.chunker import chunk_pages

    pages = load_pdfs_from_dir("data/raw")
    if not pages:
        logger.warning("No demo PDFs found in data/raw/")
        return 0

    chunks = chunk_pages(pages)
    embedder = Embedder()
    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    store.add_chunks(chunks, embeddings)
    logger.info(f"Demo ingestion complete — {len(chunks)} chunks")
    return len(chunks)


demo_chunks = ensure_demo_ingested()
chain = RAGChain()
embedder = Embedder()


# ------------------------------------------------------------
# PDF upload handler
# ------------------------------------------------------------

def ingest_pdf(file) -> str:
    """
    Ingest a user-uploaded PDF into the vector store.
    Adds to existing chunks — doesn't reset the store.
    """
    if file is None:
        return "No file uploaded."

    try:
        path = Path(file.name)
        logger.info(f"User uploaded: {path.name}")

        pages = load_pdf(path)
        if not pages:
            return f"Could not extract text from {path.name}. Is it a scanned PDF?"

        chunks = chunk_pages(pages)
        embeddings = embedder.embed_texts([c["text"] for c in chunks])

        store = VectorStore()
        store.add_chunks(chunks, embeddings)

        total = store.collection.count()
        return (
            f"✅ **{path.name}** ingested successfully!\n\n"
            f"- Pages extracted: {len(pages)}\n"
            f"- Chunks created: {len(chunks)}\n"
            f"- Total chunks in store: {total}\n\n"
            f"You can now ask questions about this document."
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return f"❌ Error ingesting file: {str(e)}"


# ------------------------------------------------------------
# Query handler
# ------------------------------------------------------------

def answer_question(question: str) -> tuple[str, str]:
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

with gr.Blocks(title="RAG Document Q&A", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # RAG Document Q&A
    Ask questions about documents using Retrieval-Augmented Generation.
    
    **Pre-loaded:** Attention Is All You Need + RAG paper (Lewis et al., 2020)
    
    **Or upload your own PDF** and query it instantly.
    """)

    with gr.Tab("Ask a question"):
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

    with gr.Tab("Upload your PDF"):
        gr.Markdown("""
        ### Upload a PDF to query
        Upload any PDF document and it will be ingested into the vector store.
        You can then ask questions about it in the **Ask a question** tab.
        
        **Note:** Uploaded documents are added to the existing store alongside the demo papers.
        Scanned PDFs (image-only) are not supported — the PDF must have extractable text.
        """)

        file_upload = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
        )
        upload_btn = gr.Button("Ingest PDF", variant="primary")
        upload_status = gr.Markdown(label="Status")

        upload_btn.click(
            fn=ingest_pdf,
            inputs=[file_upload],
            outputs=[upload_status],
        )

    gr.Markdown(
        "_Built with sentence-transformers, ChromaDB, and Groq (Llama 3.1 8B). "
        "[View source on GitHub](https://github.com/OmUniyal/rag-document-qa)_"
    )

if __name__ == "__main__":
    demo.launch()