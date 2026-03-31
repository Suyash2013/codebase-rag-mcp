"""Fallback extractor for plain text files."""

from pathlib import Path

from config.settings import settings
from mcp_server.extractors.base import ExtractionResult, ExtractorBase


class TextExtractor(ExtractorBase):
    """Fallback extractor for plain text files."""

    def supported_extensions(self) -> set[str]:
        # Respect settings.text_extensions for what we consider "text"
        return set(settings.text_extensions)

    def supported_filenames(self) -> set[str]:
        # These are handled by specific extractors if they exist,
        # but as a fallback we can accept them as plain text.
        return {
            "Dockerfile", "Makefile", "CMakeLists.txt", "Jenkinsfile",
            "Procfile", "Vagrantfile", "Gemfile", "Rakefile",
            ".gitignore", ".dockerignore", ".editorconfig",
        }

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractionResult(text=text, content_type="plain_text")
