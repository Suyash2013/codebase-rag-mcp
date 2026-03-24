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
    with patch.dict(
        os.environ,
        {
            "RAG_EMBEDDING_PROVIDER": "openai",
            "RAG_OPENAI_API_KEY": "",
        },
    ):
        from config.settings import Settings

        with (
            patch("mcp_server.embeddings.factory.settings", Settings()),
            patch("mcp_server.embeddings.openai.settings", Settings()),
        ):
            from mcp_server.embeddings.factory import create_provider

            with pytest.raises(RuntimeError, match="OpenAI API key"):
                create_provider()


def test_factory_voyage_requires_key():
    """Factory should raise when Voyage key is missing."""
    with patch.dict(
        os.environ,
        {
            "RAG_EMBEDDING_PROVIDER": "voyage",
            "RAG_VOYAGE_API_KEY": "",
        },
    ):
        from config.settings import Settings

        with (
            patch("mcp_server.embeddings.factory.settings", Settings()),
            patch("mcp_server.embeddings.voyage.settings", Settings()),
        ):
            from mcp_server.embeddings.factory import create_provider

            with pytest.raises(RuntimeError, match="Voyage API key"):
                create_provider()


def test_factory_creates_nomic():
    """Factory should create NomicProvider for 'nomic' setting (auto-pull skipped)."""
    with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "nomic"}):
        from config.settings import Settings

        mock_settings = Settings()
        with (
            patch("mcp_server.embeddings.factory.settings", mock_settings),
            patch("mcp_server.embeddings.ollama.settings", mock_settings),
            patch("mcp_server.embeddings.nomic.settings", mock_settings),
            # Skip the actual HTTP call to Ollama during __init__
            patch("mcp_server.embeddings.ollama.OllamaProvider._ensure_model_available"),
        ):
            from mcp_server.embeddings.factory import create_provider
            from mcp_server.embeddings.nomic import NomicProvider

            provider = create_provider()
            assert isinstance(provider, NomicProvider)
            assert provider._model == mock_settings.nomic_model


def test_factory_creates_snowflake():
    """Factory should create SnowflakeProvider for 'snowflake' setting (auto-pull skipped)."""
    with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "snowflake"}):
        from config.settings import Settings

        mock_settings = Settings()
        with (
            patch("mcp_server.embeddings.factory.settings", mock_settings),
            patch("mcp_server.embeddings.ollama.settings", mock_settings),
            patch("mcp_server.embeddings.snowflake.settings", mock_settings),
            # Skip the actual HTTP call to Ollama during __init__
            patch("mcp_server.embeddings.ollama.OllamaProvider._ensure_model_available"),
        ):
            from mcp_server.embeddings.factory import create_provider
            from mcp_server.embeddings.snowflake import SnowflakeProvider

            provider = create_provider()
            assert isinstance(provider, SnowflakeProvider)
            assert provider._model == mock_settings.snowflake_model


def test_ollama_auto_pull_when_model_missing():
    """OllamaProvider should pull the model if it is not available locally."""
    from unittest.mock import MagicMock

    import requests

    with patch.dict(os.environ, {}):
        from config.settings import Settings

        mock_settings = Settings()
        with patch("mcp_server.embeddings.ollama.settings", mock_settings):
            from mcp_server.embeddings.ollama import OllamaProvider

            # Simulate /api/tags returning no models
            tags_response = MagicMock()
            tags_response.json.return_value = {"models": []}
            tags_response.raise_for_status = MagicMock()

            pull_response = MagicMock()
            pull_response.raise_for_status = MagicMock()

            def fake_get(url, **kwargs):
                return tags_response

            def fake_post(url, **kwargs):
                return pull_response

            with (
                patch.object(requests, "get", side_effect=fake_get),
                patch.object(requests, "post", side_effect=fake_post),
            ):
                provider = OllamaProvider(model="nomic-embed-text:latest", auto_pull=True)
                pull_response.raise_for_status.assert_called_once()
                assert provider._model == "nomic-embed-text:latest"


def test_ollama_no_pull_when_model_present():
    """OllamaProvider should NOT pull the model if it is already available locally."""
    from unittest.mock import MagicMock

    import requests

    with patch.dict(os.environ, {}):
        from config.settings import Settings

        mock_settings = Settings()
        with patch("mcp_server.embeddings.ollama.settings", mock_settings):
            from mcp_server.embeddings.ollama import OllamaProvider

            tags_response = MagicMock()
            tags_response.json.return_value = {
                "models": [{"name": "nomic-embed-text:latest"}]
            }
            tags_response.raise_for_status = MagicMock()

            with patch.object(requests, "get", return_value=tags_response) as mock_get, patch.object(requests, "post") as mock_post:
                OllamaProvider(model="nomic-embed-text:latest", auto_pull=True)
                mock_get.assert_called_once()
                mock_post.assert_not_called()


def test_factory_unknown_provider():
    """Factory should raise for unknown provider."""
    with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "nonexistent"}):
        from config.settings import Settings

        with patch("mcp_server.embeddings.factory.settings", Settings()):
            from mcp_server.embeddings.factory import create_provider

            with pytest.raises(ValueError, match="Unknown embedding provider"):
                create_provider()
