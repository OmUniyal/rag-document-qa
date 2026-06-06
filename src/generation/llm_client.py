"""
LLM backend abstraction layer.

Supports two backends selectable via config.yaml:
  - "ollama":      Local Mistral/Llama via Ollama. Free, private, no token needed.
  - "huggingface": HuggingFace Inference API. Free tier, needs HF_API_TOKEN.

Why abstract this?
  Swapping LLM providers should NOT require changes anywhere else in the codebase.
  This is the Open/Closed principle applied to ML systems.
"""

import requests
import ollama as ollama_client
from src.utils.config import config
from src.utils.logger import logger


class LLMClient:

    def __init__(self):
        self.backend = config.llm_backend
        logger.info(f"LLM backend: {self.backend}")

    def generate(self, prompt: str) -> str:
        """
        Send a prompt, get a response string back.
        All backend differences are hidden here.
        """
        if self.backend == "ollama":
            return self._generate_ollama(prompt)
        elif self.backend == "huggingface":
            return self._generate_huggingface(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Choose 'ollama' or 'huggingface'.")

    def _generate_ollama(self, prompt: str) -> str:
        """
        Call local Ollama server.
        Ollama must be running: `ollama serve` and model pulled: `ollama pull mistral`
        """
        try:
            response = ollama_client.generate(
                model=config.ollama_model,
                prompt=prompt,
                options={
                    "temperature": config.temperature,
                    "num_predict": config.max_new_tokens,
                },
            )
            return response["response"].strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def _generate_huggingface(self, prompt: str) -> str:
        """
        Call HuggingFace Inference API.
        Free tier has rate limits. Good for testing without local GPU.
        """
        if not config.hf_api_token:
            raise ValueError("HF_API_TOKEN not set in .env")

        api_url = f"https://api-inference.huggingface.co/models/{config.hf_model_id}"
        headers = {"Authorization": f"Bearer {config.hf_api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "return_full_text": False,
            },
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        return str(result)
