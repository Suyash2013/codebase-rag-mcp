"""Tests for centralized configuration."""

import os
from unittest.mock import patch


def test_default_settings():
    """Settings should have sensible defaults."""
    with patch.dict(os.environ, {}, clear=False):
        from config.settings import Settings

        s = Settings()
        assert s.qdrant_mode == "local"
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.qdrant_collection == "documents"
        assert s.embedding_provider == "onnx"
        assert s.onnx_model_name == "all-MiniLM-L6-v2"
        assert s.ollama_embed_model == "snowflake-arctic-embed:latest"
        assert s.ingestion_timeout_hours == 24
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.default_n_results == 10
        assert s.max_n_results == 20
        assert s.max_file_size_bytes == 1_048_576
        assert "node_modules" in s.skip_directories
        assert "__pycache__" in s.skip_directories
        assert ".git" in s.skip_directories
        assert ".py" in s.text_extensions
        assert ".js" in s.text_extensions
        assert ".md" in s.text_extensions


def test_settings_from_env():
    """Settings should load from RAG_ prefixed env vars."""
    with patch.dict(
        os.environ,
        {
            "RAG_QDRANT_MODE": "remote",
            "RAG_QDRANT_HOST": "remote-host",
            "RAG_QDRANT_PORT": "6334",
            "RAG_QDRANT_COLLECTION": "my_collection",
            "RAG_EMBEDDING_PROVIDER": "ollama",
            "RAG_OLLAMA_EMBED_MODEL": "other-model:latest",
            "RAG_INGESTION_TIMEOUT_HOURS": "48",
        },
    ):
        from config.settings import Settings

        s = Settings()
        assert s.qdrant_mode == "remote"
        assert s.qdrant_host == "remote-host"
        assert s.qdrant_port == 6334
        assert s.qdrant_collection == "my_collection"
        assert s.embedding_provider == "ollama"
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


def test_get_qdrant_local_path_default():
    """get_qdrant_local_path should default to .rag-mcp/qdrant under working dir."""
    with patch.dict(
        os.environ,
        {
            "RAG_WORKING_DIRECTORY": "/my/project",
            "RAG_QDRANT_LOCAL_PATH": "",
        },
    ):
        from config.settings import Settings

        s = Settings()
        assert (
            s.get_qdrant_local_path().endswith(".rag-mcp/qdrant")
            or ".rag-mcp" in s.get_qdrant_local_path()
        )


def test_get_qdrant_local_path_override():
    """get_qdrant_local_path should use override when set."""
    with patch.dict(os.environ, {"RAG_QDRANT_LOCAL_PATH": "/custom/qdrant"}):
        from config.settings import Settings

        s = Settings()
        assert s.get_qdrant_local_path() == "/custom/qdrant"


def test_get_onnx_model_path_default():
    """get_onnx_model_path should default to .rag-mcp/models under working dir."""
    with patch.dict(
        os.environ,
        {
            "RAG_WORKING_DIRECTORY": "/my/project",
            "RAG_ONNX_MODEL_PATH": "",
        },
    ):
        from config.settings import Settings

        s = Settings()
        assert ".rag-mcp" in s.get_onnx_model_path()


def test_embedding_provider_settings():
    """Cloud provider settings should load correctly."""
    with patch.dict(
        os.environ,
        {
            "RAG_EMBEDDING_PROVIDER": "openai",
            "RAG_OPENAI_API_KEY": "sk-test-key",
            "RAG_OPENAI_EMBED_MODEL": "text-embedding-3-large",
        },
    ):
        from config.settings import Settings

        s = Settings()
        assert s.embedding_provider == "openai"
        assert s.openai_api_key == "sk-test-key"
        assert s.openai_embed_model == "text-embedding-3-large"
