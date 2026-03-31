"""Change detection base class and data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChangeReport:
    """Result of change detection."""

    has_changes: bool
    changed_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    details: str = ""


class ChangeDetector(ABC):
    """Interface for detecting file changes since last indexing."""

    @abstractmethod
    def detect_changes(self, directory: str) -> ChangeReport: ...

    @abstractmethod
    def save_checkpoint(self, directory: str) -> None: ...

    @abstractmethod
    def has_checkpoint(self, directory: str) -> bool: ...
