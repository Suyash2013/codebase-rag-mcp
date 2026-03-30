import ast
import re
from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class CodeChunker(ChunkerBase):
    """Splits code at function/class boundaries when possible."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"code"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []
        lang = (metadata or {}).get("language", "unknown")

        if lang == "python":
            blocks = self._split_python(text)
        elif lang in ("javascript", "typescript", "go", "java", "kotlin"):
            blocks = self._split_by_regex(text, lang)
        else:
            return self._fallback.chunk(text, chunk_size, chunk_overlap, metadata)

        return self._blocks_to_chunks(blocks, chunk_size, chunk_overlap, metadata)

    def _split_python(self, text: str) -> list[str]:
        """Split Python code at top-level function/class boundaries using AST."""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [text]

        lines = text.split("\n")
        blocks = []
        prev_end = 0

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                # Capture any code before this definition
                if start > prev_end:
                    preamble = "\n".join(lines[prev_end:start]).strip()
                    if preamble:
                        blocks.append(preamble)
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
                blocks.append("\n".join(lines[start:end]))
                prev_end = end

        # Remaining lines
        if prev_end < len(lines):
            remainder = "\n".join(lines[prev_end:]).strip()
            if remainder:
                blocks.append(remainder)

        return blocks if blocks else [text]

    def _split_by_regex(self, text: str, lang: str) -> list[str]:
        """Split code at function/class boundaries using regex."""
        patterns = {
            "javascript": r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\()",
            "typescript": r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|interface\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\()",
            "go": r"^(?:func\s+|type\s+\w+\s+(?:struct|interface))",
            "java": r"^(?:\s*(?:public|private|protected)\s+)?(?:static\s+)?(?:class|interface|(?:\w+\s+)+\w+\s*\()",
            "kotlin": r"^(?:fun\s+|class\s+|interface\s+|object\s+)",
        }
        pattern = patterns.get(lang)
        if not pattern:
            return [text]

        lines = text.split("\n")
        blocks = []
        current_block: list[str] = []

        for line in lines:
            if re.match(pattern, line) and current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        return [b for b in blocks if b.strip()]

    def _blocks_to_chunks(self, blocks: list[str], chunk_size: int,
                          chunk_overlap: int, metadata: dict | None) -> list[Chunk]:
        """Convert code blocks to chunks, merging small blocks and splitting large ones."""
        chunks = []
        current = ""

        for block in blocks:
            if len(current) + len(block) + 1 <= chunk_size:
                current = current + "\n" + block if current else block
            else:
                if current.strip():
                    meta = dict(metadata) if metadata else {}
                    meta["chunk_index"] = len(chunks)
                    chunks.append(Chunk(text=current.strip(), metadata=meta))
                if len(block) > chunk_size:
                    # Sub-split oversized blocks
                    sub_chunks = self._fallback.chunk(block, chunk_size, chunk_overlap, metadata)
                    for sc in sub_chunks:
                        sc.metadata["chunk_index"] = len(chunks)
                        chunks.append(sc)
                    current = ""
                else:
                    current = block

        if current.strip():
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=current.strip(), metadata=meta))

        return chunks
