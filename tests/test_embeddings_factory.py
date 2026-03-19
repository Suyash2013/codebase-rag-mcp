"""Tests for embedding provider factory."""

import os
from unittest.mock import patch

import pytest


def test_factory_creates_ollama():
    """Factory should create OllamaProvider for 'ollama' setting."""
    with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "ollama"}):
        from config.settings import Settings
        with patch("mcp_server.embeddings.factory.settings", Settings()):
            from mcp_server.embeddings.factory import create_provider
            from mcp_server.embeddings.ollama import OllamaProvider
            provider = create_provider()
            assert isinstance(provider, OllamaProvider)


def test_factory_creates_onnx():
    """Factory should create OnnxLocalProvider for 'onnx' setting."""
    with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "onnx"}):
        from config.settings import Settings
        with patch("mcp_server.embeddings.factory.settings", Settings()):
            from mcp_server.embeddings.factory import create_provider
            from mcp_server.embeddings.onnx_local import OnnxLocalProvider
            provider = create_provider()
            assert isinstance(provider, OnnxLocalProvider)


def test_factory_openai_requires_key():
    """Factory should raise when OpenAI key is missing."""
    with patch.dict(os.environ, {
        "RAG_EMBEDDING_PROVIDER": "openai",
        "RAG_OPENAI_API_KEY": "",
    }):
        from config.settings import Settings
        with patch("mcp_server.embeddings.factory.settings", Settings()):
            with patch("mcp_server.embeddings.openai.settings", Settings()):
                from mcp_server.embeddings.factory import create_provider
                with pytest.raises(RuntimeError, match="OpenAI API key"):
                    create_provider()


def test_factory_voyage_requires_key():
    """Factory should raise when Voyage key is missing."""
    with patch.dict(os.environ, {
        "RAG_EMBEDDING_PROVIDER": "voyage",
        "RAG_VOYAGE_API_KEY": "",
    }):
        from config.settings import Settings
        with patch("mcp_server.embeddings.factory.settings", Settings()):
            with patch("mcp_server.embeddings.voyage.settings", Settings()):
                from mcp_server.embeddings.factory import create_provider
                with pytest.raises(RuntimeError, match="Voyage API key"):
                    create_provider()


def test_factory_unknown_provider():
    """Factory should raise for unknown provider."""
    with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "nonexistent"}):
        from config.settings import Settings
        with patch("mcp_server.embeddings.factory.settings", Settings()):
            from mcp_server.embeddings.factory import create_provider
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                create_provider()
