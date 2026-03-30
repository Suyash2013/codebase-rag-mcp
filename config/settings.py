"""Centralized configuration for the rag-mcp server."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are configurable via environment variables with the RAG_ prefix."""

    # Qdrant
    qdrant_mode: str = "local"  # "local" (on-disk, zero-config) | "remote" (Docker/cloud)
    qdrant_local_path: str = ""  # defaults to {working_dir}/.rag-mcp/qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Embedding provider
    embedding_provider: str = "onnx"  # "onnx" | "ollama" | "openai" | "voyage"

    # ONNX local embeddings (default, zero-config)
    onnx_model_name: str = "all-MiniLM-L6-v2"
    onnx_model_path: str = ""  # defaults to {working_dir}/.rag-mcp/models/

    # Ollama embeddings
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "snowflake-arctic-embed:latest"

    # OpenAI embeddings
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"

    # Voyage AI embeddings
    voyage_api_key: str = ""
    voyage_embed_model: str = "voyage-code-3"

    # Ingestion
    ingestion_timeout_hours: int = 24
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_bytes: int = 1_048_576  # 1 MB
    skip_directories: list[str] = [
        "node_modules",
        "__pycache__",
        "venv",
        ".venv",
        "dist",
        "build",
        ".git",
        ".codebase-rag",
        ".rag-mcp",
        ".idea",
        ".vscode",
        "target",
        ".gradle",
    ]
    text_extensions: list[str] = [
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".kt",
        ".kts",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".html",
        ".css",
        ".scss",
        ".less",
        ".vue",
        ".svelte",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".xml",
        ".sql",
        ".graphql",
        ".proto",
        ".md",
        ".rst",
        ".txt",
        ".log",
        ".adoc",
        ".dockerfile",
        ".env.example",
        ".gitignore",
        ".editorconfig",
        ".gradle",
        ".cmake",
        ".makefile",
    ]

    # Search
    default_n_results: int = 10
    max_n_results: int = 20
    hybrid_search_enabled: bool = True
    hybrid_semantic_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3

    # Runtime — set to cwd at startup if not overridden
    working_directory: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    def get_working_directory(self) -> str:
        return self.working_directory or os.getcwd()

    def get_qdrant_local_path(self) -> str:
        if self.qdrant_local_path:
            return self.qdrant_local_path
        return str(Path(self.get_working_directory()) / ".rag-mcp" / "qdrant")

    def get_onnx_model_path(self) -> str:
        if self.onnx_model_path:
            return self.onnx_model_path
        return str(Path(self.get_working_directory()) / ".rag-mcp" / "models")


# Singleton instance
settings = Settings()
