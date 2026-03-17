"""Tests for Ollama embedding helper."""

import os
from unittest.mock import patch

import pytest
import responses


@pytest.fixture(autouse=True)
def reset_embed_fn():
    """Reset the cached embedding function between tests."""
    import mcp_server.embeddings as mod
    mod._embed_fn = None
    yield
    mod._embed_fn = None


@responses.activate
def test_embed_via_new_api():
    """Should use /api/embed when available."""
    fake_embedding = [0.1] * 1024
    responses.post(
        "http://localhost:11434/api/embed",
        json={"embeddings": [fake_embedding]},
        status=200,
    )

    with patch.dict(os.environ, {
        "RAG_OLLAMA_BASE_URL": "http://localhost:11434",
        "RAG_OLLAMA_EMBED_MODEL": "snowflake-arctic-embed:latest",
    }):
        # Re-import to pick up patched settings
        from config.settings import Settings
        with patch("mcp_server.embeddings.settings", Settings()):
            from mcp_server.embeddings import get_embedding
            result = get_embedding("test query")

    assert len(result) == 1024
    assert result == fake_embedding


@responses.activate
def test_embed_fallback_to_legacy():
    """Should fall back to /api/embeddings when /api/embed fails."""
    fake_embedding = [0.2] * 1024

    # New API fails
    responses.post(
        "http://localhost:11434/api/embed",
        json={"error": "not found"},
        status=404,
    )
    # Legacy API works
    responses.post(
        "http://localhost:11434/api/embeddings",
        json={"embedding": fake_embedding},
        status=200,
    )

    with patch.dict(os.environ, {
        "RAG_OLLAMA_BASE_URL": "http://localhost:11434",
        "RAG_OLLAMA_EMBED_MODEL": "snowflake-arctic-embed:latest",
    }):
        from config.settings import Settings
        with patch("mcp_server.embeddings.settings", Settings()):
            from mcp_server.embeddings import get_embedding
            result = get_embedding("test query")

    assert len(result) == 1024
    assert result == fake_embedding


@responses.activate
def test_embed_both_fail_raises():
    """Should raise RuntimeError when both APIs fail."""
    responses.post("http://localhost:11434/api/embed", status=500)
    responses.post("http://localhost:11434/api/embeddings", status=500)

    with patch.dict(os.environ, {
        "RAG_OLLAMA_BASE_URL": "http://localhost:11434",
        "RAG_OLLAMA_EMBED_MODEL": "snowflake-arctic-embed:latest",
    }):
        from config.settings import Settings
        with patch("mcp_server.embeddings.settings", Settings()):
            from mcp_server.embeddings import get_embedding
            with pytest.raises(RuntimeError, match="Could not reach Ollama"):
                get_embedding("test query")
