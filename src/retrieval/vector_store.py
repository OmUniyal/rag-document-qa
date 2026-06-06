"""
ChromaDB vector store wrapper.

ChromaDB stores: the raw text, the embedding vector, and metadata (source, page).
On disk so the vector store survives between runs — you don't re-embed on every restart.

Interview concept to own:
  Chroma uses HNSW (Hierarchical Navigable Small World) indexing under the hood.
  HNSW is an approximate nearest neighbour algorithm — it trades a tiny bit of
  recall for massive speed gains at scale. At 10M vectors, brute-force cosine
  search is unusable; HNSW is O(log n).
"""

import chromadb
from chromadb.config import Settings
from src.utils.config import config
from src.utils.logger import logger


class VectorStore:

    def __init__(self, collection_name: str = None, persist_dir: str = None):
        collection_name = collection_name or config.collection_name
        persist_dir = persist_dir or config.chroma_persist_dir

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine distance, not L2
        )
        logger.info(f"Vector store ready: '{collection_name}' at {persist_dir}")
        logger.info(f"  Current document count: {self.collection.count()}")

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        Store chunks + their embeddings in Chroma.

        Args:
            chunks:     Output from chunker.chunk_pages()
            embeddings: Parallel list of embedding vectors from Embedder.embed_texts()
        """
        if not chunks:
            logger.warning("No chunks to add.")
            return

        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        )
        logger.info(f"Added {len(chunks)} chunks. Total in store: {self.collection.count()}")

    def query(self, query_embedding: list[float], top_k: int = None) -> list[dict]:
        """
        Find the top-k most similar chunks to a query embedding.

        Returns:
            List of dicts: [{"text": str, "source": str, "page": int, "score": float}]
            Ordered by similarity (most similar first).
        """
        top_k = top_k or config.top_k
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),  # Convert cosine distance → similarity
            })

        return chunks

    def reset(self) -> None:
        """Delete and recreate the collection. Use during development to re-ingest."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{name}' has been reset.")
