"""
LLM backend — Groq API.
Fast, free tier, reliable. Uses llama-3.1-8b-instant.

Why Groq over HuggingFace inference API:
- More stable free tier
- Lower latency
- No provider routing issues
"""

import os
from groq import Groq
from src.utils.config import config
from src.utils.logger import logger


class LLMClient:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = config.groq_model
        logger.info(f"LLM client ready: Groq ({self.model})")

    def generate(self, prompt: str) -> str:
        """Send prompt, get response string back."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            raise