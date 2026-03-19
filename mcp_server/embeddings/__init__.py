"""Embedding provider package — maintains backward-compatible interface."""

from mcp_server.embeddings.base import EmbeddingProvider
from mcp_server.embeddings.factory import create_provider

_provider: EmbeddingProvider | None = None


def _get_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = create_provider()
    return _provider


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text using the configured provider."""
    return _get_provider().embed(text)


def get_embedding_dimension() -> int:
    """Get the dimension of the configured embedding model."""
    return _get_provider().dimension()


def reset_provider() -> None:
    """Reset the cached provider (for testing)."""
    global _provider
    _provider = None
