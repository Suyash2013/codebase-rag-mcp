"""Centralized configuration for the codebase-rag MCP server."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are configurable via environment variables with the RAG_ prefix."""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "codebase"

    # Ollama embeddings
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "snowflake-arctic-embed:latest"

    # Ingestion
    ingestion_timeout_hours: int = 24
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Search
    default_n_results: int = 10
    max_n_results: int = 20

    # Langflow (for bidirectional communication)
    langflow_base_url: str = "http://localhost:7860"
    langflow_flow_id: str = ""

    # Runtime — set to cwd at startup if not overridden
    working_directory: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    def get_working_directory(self) -> str:
        return self.working_directory or os.getcwd()


# Singleton instance
settings = Settings()
