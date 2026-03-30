"""Fallback extractor for plain text files."""

from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult


class TextExtractor(ExtractorBase):
    """Fallback extractor for plain text files."""

    TEXT_EXTENSIONS = {".txt", ".log", ".env.example", ".gitignore", ".editorconfig"}

    def supported_extensions(self) -> set[str]:
        return self.TEXT_EXTENSIONS

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractionResult(text=text, content_type="plain_text")
