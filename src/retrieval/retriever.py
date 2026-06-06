"""
Ties embedder + vector store into a single retrieve() call.

This is the interface the rest of the app uses.
Neither app.py nor rag_chain.py should talk to Embedder or VectorStore directly.
"""

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.utils.config import config
from src.utils.logger import logger


class Retriever:

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Given a natural language query, return the top-k relevant chunks.

        Steps:
          1. Embed the query using the same model used for documents.
             (Critical: query and document embeddings MUST use the same model.)
          2. Search the vector store for nearest neighbours.
          3. Return ranked chunks with source metadata.
        """
        top_k = top_k or config.top_k
        logger.info(f"Retrieving top-{top_k} chunks for: '{query[:60]}...'")

        query_embedding = self.embedder.embed_query(query)
        chunks = self.store.query(query_embedding, top_k=top_k)

        for i, chunk in enumerate(chunks):
            logger.debug(f"  [{i+1}] score={chunk['score']} | {chunk['source']} p{chunk['page']}")

        return chunks
