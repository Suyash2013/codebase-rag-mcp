"""Auto-registers built-in chunkers."""

from mcp_server.chunkers.base import (
    Chunk,
    ChunkerBase,
    get_chunker,
    register_chunker,
)
from mcp_server.chunkers.recursive import RecursiveChunker
from mcp_server.chunkers.code_chunker import CodeChunker
from mcp_server.chunkers.markdown_chunker import MarkdownChunker
from mcp_server.chunkers.structured_chunker import StructuredChunker
from mcp_server.chunkers.paragraph_chunker import ParagraphChunker

# Register built-in chunkers
register_chunker(RecursiveChunker())
register_chunker(CodeChunker())
register_chunker(MarkdownChunker())
register_chunker(StructuredChunker())
register_chunker(ParagraphChunker())

__all__ = [
    "Chunk", "ChunkerBase", "get_chunker", "register_chunker",
    "RecursiveChunker", "CodeChunker", "MarkdownChunker",
    "StructuredChunker", "ParagraphChunker",
]
