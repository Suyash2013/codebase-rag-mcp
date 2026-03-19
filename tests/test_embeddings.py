"""Tests for Ollama embedding provider."""

import os
from unittest.mock import patch

import pytest
import responses

from mcp_server.embeddings.ollama import OllamaProvider


@pytest.fixture(autouse=True)
def patch_settings():
    """Ensure Ollama settings are configured for tests."""
    with patch.dict(os.environ, {
        "RAG_OLLAMA_BASE_URL": "http://localhost:11434",
        "RAG_OLLAMA_EMBED_MODEL": "snowflake-arctic-embed:latest",
        "RAG_EMBEDDING_PROVIDER": "ollama",
    }):
        yield


@responses.activate
def test_embed_via_new_api():
    """Should use /api/embed when available."""
    fake_embedding = [0.1] * 1024
    responses.post(
        "http://localhost:11434/api/embed",
        json={"embeddings": [fake_embedding]},
        status=200,
    )

    provider = OllamaProvider()
    result = provider.embed("test query")

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

    provider = OllamaProvider()
    result = provider.embed("test query")

    assert len(result) == 1024
    assert result == fake_embedding


@responses.activate
def test_embed_both_fail_raises():
    """Should raise RuntimeError when both APIs fail."""
    responses.post("http://localhost:11434/api/embed", status=500)
    responses.post("http://localhost:11434/api/embeddings", status=500)

    provider = OllamaProvider()
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        provider.embed("test query")
