"""
Gradio UI — the user-facing demo app.
Deploy to HuggingFace Spaces by pushing this file + requirements.txt.

Run locally:
    python app.py
"""

import gradio as gr
from src.generation.rag_chain import RAGChain
from src.utils.logger import logger

chain = RAGChain()


def answer_question(question: str) -> tuple[str, str]:
    """Gradio callback: takes question, returns (answer, sources_markdown)."""
    if not question.strip():
        return "Please enter a question.", ""

    result = chain.query(question)

    answer = result["answer"]

    # Format sources as a readable table
    sources_lines = ["**Sources retrieved:**\n"]
    for i, s in enumerate(result["sources"], start=1):
        sources_lines.append(
            f"{i}. `{s['source']}` — Page {s['page']} (similarity: {s['score']})"
        )
    sources_md = "\n".join(sources_lines)

    return answer, sources_md


with gr.Blocks(title="RAG Document Q&A") as demo:
    gr.Markdown("## RAG Document Q&A\nAsk questions about your ingested documents.")

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="What does the document say about...?",
            lines=2,
        )

    submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
        sources_box = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box],
        outputs=[answer_box, sources_box],
    )

    gr.Markdown(
        "_Note: Answers are grounded only in ingested documents. "
        "Run `python ingest.py` first to load your PDFs._"
    )

if __name__ == "__main__":
    demo.launch(share=False)
