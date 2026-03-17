"""Ollama embedding helper — supports both /api/embed (new) and /api/embeddings (legacy)."""

import logging

import requests

from config.settings import settings

log = logging.getLogger("codebase-rag-mcp")


def _embed_via_new_api(text: str) -> list[float]:
    """Ollama >= 0.4: POST /api/embed  {model, input} -> {embeddings: [[...]]}"""
    resp = requests.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.ollama_embed_model, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _embed_via_legacy_api(text: str) -> list[float]:
    """Ollama < 0.4: POST /api/embeddings  {model, prompt} -> {embedding: [...]}"""
    resp = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# Auto-detect which endpoint works on first call, then stick with it.
_embed_fn = None


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text using Ollama's snowflake-arctic-embed model."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn(text)

    # Try new API first
    try:
        result = _embed_via_new_api(text)
        _embed_fn = _embed_via_new_api
        log.info("Using Ollama /api/embed (new endpoint)")
        return result
    except Exception:
        pass

    # Fall back to legacy
    try:
        result = _embed_via_legacy_api(text)
        _embed_fn = _embed_via_legacy_api
        log.info("Using Ollama /api/embeddings (legacy endpoint)")
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            f"Ensure Ollama is running and '{settings.ollama_embed_model}' is pulled."
        ) from exc


def get_embedding_dimension() -> int:
    """Get the dimension of the embedding model by embedding a test string."""
    test_embedding = get_embedding("dimension test")
    return len(test_embedding)
