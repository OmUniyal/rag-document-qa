"""
The complete RAG pipeline in one place.

RAGChain.query() is the single entrypoint for the app and API.
It orchestrates: retrieve → build prompt → generate → return with sources.

This is the class you demo in interviews.
"""

from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import LLMClient
from src.utils.config import config
from src.utils.logger import logger


class RAGChain:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMClient()

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Full RAG pipeline: question in, answer + sources out.

        Returns:
            {
                "question": str,
                "answer":   str,
                "sources":  [{"source": str, "page": int, "score": float}],
                "chunks_used": int,
            }
        """
        logger.info(f"Query received: '{question}'")

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, top_k=top_k or config.top_k)

        # Step 2: Build the grounded prompt
        prompt = build_prompt(question, chunks)

        # Step 3: Generate answer
        logger.info("Sending to LLM...")
        answer = self.llm.generate(prompt)

        # Step 4: Package sources for attribution
        sources = [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in chunks
        ]

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
        }

        logger.info(f"Answer generated. Sources: {[s['source'] for s in sources]}")
        return result
