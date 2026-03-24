"""Snowflake embedding provider — uses snowflake-arctic-embed via Ollama with auto-pull."""

from config.settings import settings
from mcp_server.embeddings.ollama import OllamaProvider


class SnowflakeProvider(OllamaProvider):
    """Embedding provider using Snowflake's snowflake-arctic-embed model via Ollama.

    If the model is not present locally it is pulled automatically on first use.
    """

    def __init__(self) -> None:
        super().__init__(model=settings.snowflake_model, auto_pull=True)
