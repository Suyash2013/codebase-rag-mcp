"""Chunker base class and registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of content ready for embedding."""

    text: str
    metadata: dict = field(default_factory=dict)


class ChunkerBase(ABC):
    """Interface for content chunkers."""

    @abstractmethod
    def content_types(self) -> set[str]:
        """Which content_types this chunker handles."""
        ...

    @abstractmethod
    def chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into chunks."""
        ...


_chunker_registry: dict[str, ChunkerBase] = {}


def register_chunker(chunker: ChunkerBase) -> None:
    for ct in chunker.content_types():
        _chunker_registry[ct] = chunker


def get_chunker(content_type: str) -> ChunkerBase:
    chunker = _chunker_registry.get(content_type) or _chunker_registry.get("plain_text")
    if chunker is None:
        from mcp_server.chunkers.recursive import RecursiveChunker

        return RecursiveChunker()
    return chunker
