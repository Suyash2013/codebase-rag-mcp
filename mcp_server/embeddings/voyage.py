"""Voyage AI embedding provider."""

import logging

import requests

from config.settings import settings
from mcp_server.embeddings.base import EmbeddingProvider

log = logging.getLogger("omni-rag")

# Known dimensions for Voyage models
_KNOWN_DIMENSIONS = {
    "voyage-code-3": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-2": 1536,
}


class VoyageProvider(EmbeddingProvider):
    """Embedding provider using Voyage AI's API."""

    def __init__(self) -> None:
        if not settings.voyage_api_key:
            raise RuntimeError("Voyage API key not configured. Set RAG_VOYAGE_API_KEY.")

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.voyage_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.voyage_embed_model,
                "input": text,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def dimension(self) -> int:
        dim = _KNOWN_DIMENSIONS.get(settings.voyage_embed_model)
        if dim:
            return dim
        return len(self.embed("dimension test"))
