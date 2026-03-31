from pathlib import Path

from mcp_server.extractors.base import ExtractionResult, ExtractorBase

_FORMAT_MAP = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".csv": "csv",
    ".tsv": "csv",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
}


class StructuredDataExtractor(ExtractorBase):
    """Extractor for structured data formats."""

    def supported_extensions(self) -> set[str]:
        return set(_FORMAT_MAP.keys())

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        fmt = _FORMAT_MAP.get(path.suffix.lower(), "unknown")
        return ExtractionResult(
            text=text,
            content_type="structured_data",
            metadata={"format": fmt},
        )
