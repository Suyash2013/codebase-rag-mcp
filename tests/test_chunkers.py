"""Tests for chunker base and RecursiveChunker."""

from mcp_server.chunkers.base import Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class TestRecursiveChunker:
    def setup_method(self):
        self.chunker = RecursiveChunker()

    def test_content_types(self):
        assert "plain_text" in self.chunker.content_types()

    def test_small_text_single_chunk(self):
        chunks = self.chunker.chunk("hello world", chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert isinstance(chunks[0], Chunk)

    def test_empty_text(self):
        chunks = self.chunker.chunk("", chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 0

    def test_splits_long_text(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        chunks = self.chunker.chunk(text, chunk_size=20, chunk_overlap=5)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_metadata_passed_through(self):
        chunks = self.chunker.chunk(
            "hello", chunk_size=1000, chunk_overlap=0,
            metadata={"file_path": "test.txt"}
        )
        assert chunks[0].metadata["file_path"] == "test.txt"
