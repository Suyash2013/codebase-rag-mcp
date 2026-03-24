"""Embedding provider factory."""

import logging

from config.settings import settings
from mcp_server.embeddings.base import EmbeddingProvider

log = logging.getLogger("codebase-rag-mcp")


def create_provider() -> EmbeddingProvider:
    """Create an embedding provider based on settings."""
    provider_name = settings.embedding_provider

    match provider_name:
        case "onnx":
            from mcp_server.embeddings.onnx_local import OnnxLocalProvider

            return OnnxLocalProvider()
        case "ollama":
            from mcp_server.embeddings.ollama import OllamaProvider

            return OllamaProvider()
        case "openai":
            from mcp_server.embeddings.openai import OpenAIProvider

            return OpenAIProvider()
        case "voyage":
            from mcp_server.embeddings.voyage import VoyageProvider

            return VoyageProvider()
        case "nomic":
            from mcp_server.embeddings.nomic import NomicProvider

            return NomicProvider()
        case "snowflake":
            from mcp_server.embeddings.snowflake import SnowflakeProvider

            return SnowflakeProvider()
        case _:
            raise ValueError(
                f"Unknown embedding provider '{provider_name}'. "
                f"Choose from: onnx, ollama, openai, voyage, nomic, snowflake"
            )
