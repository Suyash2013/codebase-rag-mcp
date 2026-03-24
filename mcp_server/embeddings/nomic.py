"""Nomic embedding provider — uses nomic-embed-text via Ollama with auto-pull."""

from config.settings import settings
from mcp_server.embeddings.ollama import OllamaProvider


class NomicProvider(OllamaProvider):
    """Embedding provider using Nomic's nomic-embed-text model via Ollama.

    If the model is not present locally it is pulled automatically on first use.
    """

    def __init__(self) -> None:
        super().__init__(model=settings.nomic_model, auto_pull=True)
