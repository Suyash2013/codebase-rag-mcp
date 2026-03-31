"""Recursive character text splitter — default fallback chunker."""

from typing import ClassVar

from mcp_server.chunkers.base import Chunk, ChunkerBase


class RecursiveChunker(ChunkerBase):
    """Recursive character text splitter. Default fallback chunker."""

    SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", " ", ""]

    def content_types(self) -> set[str]:
        return {"plain_text"}

    def chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []
        raw_chunks = self._split(text, chunk_size, self.SEPARATORS)
        # Apply overlap
        result = []
        for i, chunk_text in enumerate(raw_chunks):
            if i > 0 and chunk_overlap > 0:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-chunk_overlap:]
                chunk_text = overlap_text + chunk_text
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = i
            result.append(Chunk(text=chunk_text, metadata=meta))
        return result

    def _split(self, text: str, chunk_size: int, separators: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            # Hard split at chunk_size
            chunks = []
            for i in range(0, len(text), chunk_size):
                piece = text[i : i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            return chunks

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    chunks.extend(self._split(part, chunk_size, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks
