"""Ollama embedding provider — supports both /api/embed (new) and /api/embeddings (legacy)."""

import logging
from collections.abc import Callable

import requests

from config.settings import settings
from mcp_server.embeddings.base import EmbeddingProvider

log = logging.getLogger("codebase-rag-mcp")


class OllamaProvider(EmbeddingProvider):
    """Embedding provider using Ollama's local API."""

    def __init__(self) -> None:
        self._embed_fn: Callable[[str], list[float]] | None = None

    def _embed_via_new_api(self, text: str) -> list[float]:
        """Ollama >= 0.4: POST /api/embed  {model, input} -> {embeddings: [[...]]}"""
        resp = requests.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.ollama_embed_model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def _embed_via_legacy_api(self, text: str) -> list[float]:
        """Ollama < 0.4: POST /api/embeddings  {model, prompt} -> {embedding: [...]}"""
        resp = requests.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.ollama_embed_model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed(self, text: str) -> list[float]:
        if self._embed_fn is not None:
            return self._embed_fn(text)

        # Try new API first
        try:
            result = self._embed_via_new_api(text)
            self._embed_fn = self._embed_via_new_api
            log.info("Using Ollama /api/embed (new endpoint)")
            return result
        except Exception:
            pass

        # Fall back to legacy
        try:
            result = self._embed_via_legacy_api(text)
            self._embed_fn = self._embed_via_legacy_api
            log.info("Using Ollama /api/embeddings (legacy endpoint)")
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_base_url}. "
                f"Ensure Ollama is running and '{settings.ollama_embed_model}' is pulled."
            ) from exc

    def dimension(self) -> int:
        return len(self.embed("dimension test"))
