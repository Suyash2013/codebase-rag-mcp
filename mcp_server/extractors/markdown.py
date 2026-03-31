from pathlib import Path

from mcp_server.extractors.base import ExtractionResult, ExtractorBase


class MarkdownExtractor(ExtractorBase):
    """Extractor for markdown and similar document formats."""

    def supported_extensions(self) -> set[str]:
        return {".md", ".rst", ".adoc"}

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractionResult(
            text=text,
            content_type="markdown",
            metadata={"format": path.suffix.lstrip(".")},
        )
