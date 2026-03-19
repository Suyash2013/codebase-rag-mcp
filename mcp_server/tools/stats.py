"""MCP tool for collection statistics."""

import json

from mcp_server.qdrant_client import get_stats


def collection_stats() -> str:
    """Get statistics about the indexed codebase.

    Returns chunk count, collection name, storage mode, and embedding provider.
    Use this to verify the index is loaded and check its size.
    """
    try:
        stats = get_stats()
        return json.dumps(stats, indent=2)
    except Exception as exc:
        return f"Error getting collection stats: {exc}"
