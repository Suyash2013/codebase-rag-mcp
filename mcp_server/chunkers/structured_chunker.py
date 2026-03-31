import json

from mcp_server.chunkers.base import Chunk, ChunkerBase
from mcp_server.chunkers.recursive import RecursiveChunker


class StructuredChunker(ChunkerBase):
    """Splits structured data (JSON, YAML, CSV) at logical boundaries."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"structured_data"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []

        fmt = (metadata or {}).get("format", "").lower()

        if fmt == "csv":
            return self._chunk_csv(text, chunk_size, metadata)
        elif fmt == "json":
            return self._chunk_json(text, chunk_size, metadata)
        else:
            return self._fallback.chunk(text, chunk_size, chunk_overlap, metadata)

    def _chunk_csv(self, text: str, chunk_size: int, metadata: dict | None) -> list[Chunk]:
        """Split CSV by row groups, preserving header in each chunk."""
        lines = text.strip().split("\n")
        if not lines:
            return []

        header = lines[0]
        rows = lines[1:]
        chunks: list[Chunk] = []
        current_rows: list[str] = []
        current_size = len(header)

        for row in rows:
            if current_size + len(row) + 1 > chunk_size and current_rows:
                chunk_text = header + "\n" + "\n".join(current_rows)
                meta = dict(metadata) if metadata else {}
                meta["chunk_index"] = len(chunks)
                chunks.append(Chunk(text=chunk_text, metadata=meta))
                current_rows = []
                current_size = len(header)
            current_rows.append(row)
            current_size += len(row) + 1

        if current_rows:
            chunk_text = header + "\n" + "\n".join(current_rows)
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=chunk_text, metadata=meta))

        return chunks

    def _chunk_json(self, text: str, chunk_size: int, metadata: dict | None) -> list[Chunk]:
        """Split JSON at top-level keys."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._fallback.chunk(text, chunk_size, 0, metadata)

        if not isinstance(data, dict):
            return self._fallback.chunk(text, chunk_size, 0, metadata)

        chunks: list[Chunk] = []
        current_obj: dict = {}
        current_size = 2  # {}

        for key, value in data.items():
            entry_str = json.dumps({key: value})
            entry_size = len(entry_str)

            if current_size + entry_size > chunk_size and current_obj:
                meta = dict(metadata) if metadata else {}
                meta["chunk_index"] = len(chunks)
                chunks.append(Chunk(text=json.dumps(current_obj, indent=2), metadata=meta))
                current_obj = {}
                current_size = 2

            current_obj[key] = value
            current_size += entry_size

        if current_obj:
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=json.dumps(current_obj, indent=2), metadata=meta))

        return chunks
