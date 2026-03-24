"""Ollama embedding provider — supports both /api/embed (new) and /api/embeddings (legacy)."""

import logging
from collections.abc import Callable

import requests

from config.settings import settings
from mcp_server.embeddings.base import EmbeddingProvider

log = logging.getLogger("codebase-rag-mcp")


class OllamaProvider(EmbeddingProvider):
    """Embedding provider using Ollama's local API.

    Args:
        model: Override the model name (defaults to settings.ollama_embed_model).
        auto_pull: If True, automatically pull the model from Ollama if not present locally.
    """

    def __init__(self, model: str | None = None, auto_pull: bool = False) -> None:
        self._model = model or settings.ollama_embed_model
        self._auto_pull = auto_pull
        self._embed_fn: Callable[[str], list[float]] | None = None
        if auto_pull:
            self._ensure_model_available()

    # ------------------------------------------------------------------
    # Auto-pull helpers
    # ------------------------------------------------------------------

    def _list_local_models(self) -> list[str]:
        """Return the names of all models currently available in Ollama."""
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=30)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def _pull_model(self) -> None:
        """Pull the model from Ollama's registry (blocking, no streaming)."""
        log.info("Auto-pulling Ollama model '%s' — this may take a while…", self._model)
        resp = requests.post(
            f"{settings.ollama_base_url}/api/pull",
            json={"model": self._model, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        log.info("Model '%s' pulled successfully.", self._model)

    def _ensure_model_available(self) -> None:
        """Pull the model if it is not already available locally."""
        try:
            local_models = self._list_local_models()
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_base_url} to check for "
                f"model '{self._model}'. Ensure Ollama is running."
            ) from exc

        # Compare by base name to handle tag variants (e.g. "nomic-embed-text:latest")
        model_base = self._model.split(":")[0]
        already_present = any(m.split(":")[0] == model_base for m in local_models)
        if not already_present:
            self._pull_model()

    # ------------------------------------------------------------------
    # Embed helpers
    # ------------------------------------------------------------------

    def _embed_via_new_api(self, text: str) -> list[float]:
        """Ollama >= 0.4: POST /api/embed  {model, input} -> {embeddings: [[...]]}"""
        resp = requests.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": self._model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def _embed_via_legacy_api(self, text: str) -> list[float]:
        """Ollama < 0.4: POST /api/embeddings  {model, prompt} -> {embedding: [...]}"""
        resp = requests.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
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
                f"Ensure Ollama is running and '{self._model}' is pulled."
            ) from exc

    def dimension(self) -> int:
        return len(self.embed("dimension test"))
