"""
Loads config.yaml and .env into a single typed config object.
Everything in the codebase imports from here — no hardcoded values elsewhere.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Single source of truth for all runtime settings."""

    def __init__(self):
        cfg = _load_yaml()

        # Ingestion
        self.chunk_size: int = cfg["ingestion"]["chunk_size"]
        self.chunk_overlap: int = cfg["ingestion"]["chunk_overlap"]

        # Retrieval
        self.embedding_model: str = cfg["retrieval"]["embedding_model"]
        self.top_k: int = cfg["retrieval"]["top_k"]
        self.collection_name: str = cfg["retrieval"]["collection_name"]
        self.chroma_persist_dir: str = cfg["retrieval"]["chroma_persist_dir"]

        # Generation
        self.llm_backend: str = cfg["generation"]["backend"]
        self.ollama_model: str = cfg["generation"]["ollama_model"]
        self.hf_model_id: str = cfg["generation"]["hf_model_id"]
        self.max_new_tokens: int = cfg["generation"]["max_new_tokens"]
        self.temperature: float = cfg["generation"]["temperature"]
        self.groq_model: str = cfg["generation"]["groq_model"]

        # Secrets from .env (never from yaml)
        self.hf_api_token: str = os.getenv("HF_API_TOKEN", "")
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Evaluation
        self.faithfulness_threshold: float = cfg["evaluation"]["faithfulness_threshold"]

    def __repr__(self):
        return (
            f"Config(backend={self.llm_backend}, "
            f"embedding={self.embedding_model}, "
            f"chunk_size={self.chunk_size})"
        )


# Module-level singleton — import this everywhere
config = Config()
