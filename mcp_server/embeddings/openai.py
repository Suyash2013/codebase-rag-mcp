"""OpenAI-compatible embedding provider."""

import logging

import requests

from config.settings import settings
from mcp_server.embeddings.base import EmbeddingProvider

log = logging.getLogger("rag-mcp")

# Known dimensions for common models
_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key not configured. Set RAG_OPENAI_API_KEY.")

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_embed_model,
                "input": text,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def dimension(self) -> int:
        dim = _KNOWN_DIMENSIONS.get(settings.openai_embed_model)
        if dim:
            return dim
        return len(self.embed("dimension test"))
