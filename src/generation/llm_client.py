"""
LLM backend — Ollama (self-hosted) with Groq fallback.
Primary: phi3:mini on nitro-server via Tailscale.
Fallback: Groq API (llama-3.1-8b-instant) if Ollama is unreachable.

Why this pattern:
- Self-hosted first: no API costs, no rate limits
- Groq fallback: ensures availability if server is down
- Same interface: rest of codebase is unaware of which backend is used
"""

import os
import requests
from groq import Groq
from src.utils.config import config
from src.utils.logger import logger

OLLAMA_URL = "http://nitro-server:11434/api/generate"


class LLMClient:

    def __init__(self):
        self.model_ollama = "phi3:mini"
        self.ollama_url = OLLAMA_URL

        # Groq as fallback
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        self.groq_model = config.groq_model

        logger.info(f"LLM client ready: Ollama primary ({self.model_ollama}), Groq fallback ({'enabled' if self.groq_client else 'disabled'})")

    def generate(self, prompt: str) -> str:
        """Try Ollama first, fall back to Groq if unavailable."""
        try:
            return self._ollama_generate(prompt)
        except Exception as e:
            logger.warning(f"Ollama unavailable: {e}. Falling back to Groq...")
            if self.groq_client:
                return self._groq_generate(prompt)
            raise RuntimeError("Both Ollama and Groq are unavailable.") from e

    def _ollama_generate(self, prompt: str) -> str:
        """Send prompt to self-hosted Ollama."""
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_ollama,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_new_tokens,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def _groq_generate(self, prompt: str) -> str:
        """Send prompt to Groq API."""
        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        return completion.choices[0].message.content.strip()