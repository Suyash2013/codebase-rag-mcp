"""Tests for centralized configuration."""

import os
from unittest.mock import patch

import pytest


def test_default_settings():
    """Settings should have sensible defaults."""
    with patch.dict(os.environ, {}, clear=False):
        from config.settings import Settings
        s = Settings()
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.qdrant_collection == "codebase"
        assert s.ollama_embed_model == "snowflake-arctic-embed:latest"
        assert s.ingestion_timeout_hours == 24
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.default_n_results == 10
        assert s.max_n_results == 20


def test_settings_from_env():
    """Settings should load from RAG_ prefixed env vars."""
    with patch.dict(os.environ, {
        "RAG_QDRANT_HOST": "remote-host",
        "RAG_QDRANT_PORT": "6334",
        "RAG_QDRANT_COLLECTION": "my_collection",
        "RAG_OLLAMA_EMBED_MODEL": "other-model:latest",
        "RAG_INGESTION_TIMEOUT_HOURS": "48",
    }):
        from config.settings import Settings
        s = Settings()
        assert s.qdrant_host == "remote-host"
        assert s.qdrant_port == 6334
        assert s.qdrant_collection == "my_collection"
        assert s.ollama_embed_model == "other-model:latest"
        assert s.ingestion_timeout_hours == 48


def test_get_working_directory_default():
    """get_working_directory should return cwd when not set."""
    with patch.dict(os.environ, {"RAG_WORKING_DIRECTORY": ""}):
        from config.settings import Settings
        s = Settings()
        assert s.get_working_directory() == os.getcwd()


def test_get_working_directory_override():
    """get_working_directory should return override when set."""
    with patch.dict(os.environ, {"RAG_WORKING_DIRECTORY": "/custom/path"}):
        from config.settings import Settings
        s = Settings()
        assert s.get_working_directory() == "/custom/path"
