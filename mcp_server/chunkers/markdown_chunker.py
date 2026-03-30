import re
from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class MarkdownChunker(ChunkerBase):
    """Splits markdown at heading boundaries."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"markdown"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []

        sections = self._split_by_headings(text)
        chunks = []

        for title, content in sections:
            section_text = content.strip()
            if not section_text:
                continue

            if len(section_text) <= chunk_size:
                meta = dict(metadata) if metadata else {}
                meta["chunk_index"] = len(chunks)
                if title:
                    meta["section_title"] = title
                chunks.append(Chunk(text=section_text, metadata=meta))
            else:
                # Sub-split oversized sections
                sub_meta = dict(metadata) if metadata else {}
                if title:
                    sub_meta["section_title"] = title
                sub_chunks = self._fallback.chunk(section_text, chunk_size, chunk_overlap, sub_meta)
                for sc in sub_chunks:
                    sc.metadata["chunk_index"] = len(chunks)
                    chunks.append(sc)

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str | None, str]]:
        """Split text into (heading_title, content) pairs."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = []
        last_end = 0
        last_title = None

        for match in heading_pattern.finditer(text):
            if match.start() > last_end:
                content = text[last_end:match.start()]
                if content.strip():
                    sections.append((last_title, content))
            last_title = match.group(2).strip()
            last_end = match.end()

        # Remaining content
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip():
                sections.append((last_title, remaining))
        elif last_title and not sections:
            sections.append((last_title, text))

        if not sections and text.strip():
            sections.append((None, text))

        return sections
