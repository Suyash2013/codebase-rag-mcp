"""MCP tool for collection statistics."""

import json

from config.settings import settings
from mcp_server.qdrant_client import get_stats


def collection_stats() -> str:
    """Get statistics about the indexed codebase.

    Returns chunk count, collection name, Qdrant host, and embedding model.
    Use this to verify the index is loaded and check its size.
    """
    try:
        stats = get_stats()
        stats["ollama_url"] = settings.ollama_base_url
        return json.dumps(stats, indent=2)
    except Exception as exc:
        return f"Error getting collection stats: {exc}"
