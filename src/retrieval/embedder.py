"""
Converts text (chunks or queries) into dense vector embeddings.

Why sentence-transformers and not the OpenAI embeddings API?
- Runs fully locally — no cost, no rate limits, no data leaving your machine.
- You understand what the model IS: a fine-tuned BERT encoder.
- In interviews: "I used sentence-transformers because I wanted full control
  over the embedding layer and to avoid vendor lock-in."

Model choice:
- all-MiniLM-L6-v2: fast, small (80MB), good general quality. Good for dev.
- bge-large-en-v1.5: slower, larger, better retrieval quality. Good for prod.
"""

from sentence_transformers import SentenceTransformer
from src.utils.config import config
from src.utils.logger import logger


class Embedder:
    """Wraps a sentence-transformer model with a consistent interface."""

    def __init__(self, model_name: str = None):
        model_name = model_name or config.embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"  Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings.

        Returns list of float vectors (one per input string).
        Batching is important: embedding 1000 texts one-by-one is ~10x slower
        than batching them together.
        """
        if not texts:
            return []
        logger.debug(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Kept separate from embed_texts because some models use different
        pooling for queries vs documents (asymmetric embedding models).
        """
        return self.model.encode(query).tolist()
