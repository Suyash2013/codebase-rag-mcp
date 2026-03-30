"""Auto-registers built-in chunkers."""

from mcp_server.chunkers.base import (
    Chunk,
    ChunkerBase,
    get_chunker,
    register_chunker,
)
from mcp_server.chunkers.recursive import RecursiveChunker

# Register built-in chunkers
register_chunker(RecursiveChunker())

__all__ = ["Chunk", "ChunkerBase", "get_chunker", "register_chunker", "RecursiveChunker"]
