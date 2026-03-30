"""Extractor base class and registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionResult:
    """Result of extracting content from a file."""
    text: str
    content_type: str  # "code", "markdown", "structured_data", "document", "plain_text"
    metadata: dict = field(default_factory=dict)


class ExtractorBase(ABC):
    """Interface for file content extractors."""

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        ...

    @abstractmethod
    def supported_filenames(self) -> set[str]:
        ...

    def can_extract(self, path: Path) -> bool:
        if path.name in self.supported_filenames():
            return True
        return path.suffix.lower() in self.supported_extensions()

    @abstractmethod
    def extract(self, path: Path) -> ExtractionResult:
        ...

    def max_file_size(self) -> int:
        return 1_000_000


_registry: dict[str, ExtractorBase] = {}
_fallback: ExtractorBase | None = None


def register_extractor(extractor: ExtractorBase, is_fallback: bool = False) -> None:
    global _fallback
    for ext in extractor.supported_extensions():
        _registry[ext] = extractor
    for name in extractor.supported_filenames():
        _registry[name] = extractor
    if is_fallback:
        _fallback = extractor


def get_extractor(path: Path) -> ExtractorBase | None:
    if path.name in _registry:
        return _registry[path.name]
    if path.suffix.lower() in _registry:
        return _registry[path.suffix.lower()]
    return _fallback
