from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class ParagraphChunker(ChunkerBase):
    """Splits document content at paragraph boundaries."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"document"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            candidate = current + "\n\n" + para if current else para
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    meta = dict(metadata) if metadata else {}
                    meta["chunk_index"] = len(chunks)
                    chunks.append(Chunk(text=current, metadata=meta))
                if len(para) > chunk_size:
                    sub_chunks = self._fallback.chunk(para, chunk_size, chunk_overlap, metadata)
                    for sc in sub_chunks:
                        sc.metadata["chunk_index"] = len(chunks)
                        chunks.append(sc)
                    current = ""
                else:
                    current = para

        if current:
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=current, metadata=meta))

        return chunks
